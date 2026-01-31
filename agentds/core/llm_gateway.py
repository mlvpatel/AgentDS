"""
Personal Data Scientist - LLM Gateway.

Universal gateway for 100+ LLM providers via LiteLLM with Logfire observability.

Author: Malav Patel
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import litellm
from litellm import acompletion, completion
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentds.core.config import Settings, get_settings
from agentds.core.logger import get_logger, get_logfire_span, log_llm_call

logger = get_logger(__name__)


class LLMMessage(BaseModel):
    """Message for LLM conversation."""

    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str = Field(..., description="Response content")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    usage: Dict[str, int] = Field(
        default_factory=dict, description="Token usage statistics"
    )
    cost: float = Field(default=0.0, description="Estimated cost in USD")
    latency_ms: float = Field(default=0.0, description="Response latency in ms")


@dataclass
class FallbackChain:
    """Fallback chain configuration."""

    models: List[str] = field(default_factory=list)
    current_index: int = 0

    def get_current_model(self) -> Optional[str]:
        """Get current model in chain."""
        if self.current_index < len(self.models):
            return self.models[self.current_index]
        return None

    def advance(self) -> bool:
        """Move to next model in chain. Returns False if exhausted."""
        self.current_index += 1
        return self.current_index < len(self.models)

    def reset(self) -> None:
        """Reset chain to first model."""
        self.current_index = 0


class LLMGateway:
    """
    Universal LLM Gateway via LiteLLM.

    Provides unified interface to 100+ LLM providers with:
    - Automatic fallback on failures
    - Response caching
    - Cost tracking
    - Logfire observability
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize LLM Gateway.

        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._configure_litellm()
        self._load_config()
        self._total_cost = 0.0

    def _configure_litellm(self) -> None:
        """Configure LiteLLM with API keys and settings."""
        llm_settings = self.settings.llm

        # Set API keys as environment variables for LiteLLM
        if llm_settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = llm_settings.openai_api_key
        if llm_settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = llm_settings.anthropic_api_key
        if llm_settings.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = llm_settings.gemini_api_key
        if llm_settings.groq_api_key:
            os.environ["GROQ_API_KEY"] = llm_settings.groq_api_key
        if llm_settings.mistral_api_key:
            os.environ["MISTRAL_API_KEY"] = llm_settings.mistral_api_key
        if llm_settings.together_api_key:
            os.environ["TOGETHERAI_API_KEY"] = llm_settings.together_api_key
        if llm_settings.deepseek_api_key:
            os.environ["DEEPSEEK_API_KEY"] = llm_settings.deepseek_api_key
        if llm_settings.xai_api_key:
            os.environ["XAI_API_KEY"] = llm_settings.xai_api_key
        if llm_settings.cohere_api_key:
            os.environ["COHERE_API_KEY"] = llm_settings.cohere_api_key
        if llm_settings.azure_api_key:
            os.environ["AZURE_API_KEY"] = llm_settings.azure_api_key
        if llm_settings.azure_api_base:
            os.environ["AZURE_API_BASE"] = llm_settings.azure_api_base
        if llm_settings.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = llm_settings.aws_access_key_id
        if llm_settings.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = llm_settings.aws_secret_access_key
        os.environ["AWS_REGION_NAME"] = llm_settings.aws_region_name

        # Configure LiteLLM settings
        litellm.drop_params = True
        litellm.set_verbose = self.settings.debug

    def _load_config(self) -> None:
        """Load LLM configuration from YAML."""
        self.config = self.settings.get_llm_config()
        self.agent_mapping = self.config.get("agent_llm_mapping", {})
        self.fallback_chains = self.config.get("fallback_chains", {})

    def get_model_for_agent(self, agent_name: str) -> str:
        """
        Get configured model for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Model identifier string
        """
        agent_config = self.agent_mapping.get(agent_name, {})
        return agent_config.get("model", self.settings.llm.default_model)

    def get_temperature_for_agent(self, agent_name: str) -> float:
        """
        Get configured temperature for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Temperature value
        """
        agent_config = self.agent_mapping.get(agent_name, {})
        return agent_config.get("temperature", self.settings.llm.default_temperature)

    def get_fallback_chain(self, chain_name: str = "default") -> FallbackChain:
        """
        Get fallback chain by name.

        Args:
            chain_name: Name of the fallback chain

        Returns:
            FallbackChain instance
        """
        models = self.fallback_chains.get(chain_name, [])
        if not models:
            models = self.fallback_chains.get("default", [self.settings.llm.default_model])
        return FallbackChain(models=models)

    @retry(
        retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def complete(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion from LLM with Logfire tracing.

        Args:
            messages: List of messages for the conversation
            model: Model to use (overrides agent config)
            temperature: Temperature setting (overrides agent config)
            max_tokens: Maximum tokens to generate
            agent_name: Agent name for config lookup
            use_fallback: Whether to use fallback chain on failure
            **kwargs: Additional parameters for LiteLLM

        Returns:
            LLMResponse with generated content
        """
        # Determine model and temperature
        if model is None and agent_name:
            model = self.get_model_for_agent(agent_name)
        model = model or self.settings.llm.default_model

        if temperature is None and agent_name:
            temperature = self.get_temperature_for_agent(agent_name)
        temperature = temperature if temperature is not None else self.settings.llm.default_temperature

        # Convert messages to dict format
        msg_list = [
            {"role": m.role, "content": m.content} if isinstance(m, LLMMessage) else m
            for m in messages
        ]

        # Setup fallback chain
        fallback = self.get_fallback_chain() if use_fallback else FallbackChain(models=[model])
        fallback.models.insert(0, model)  # Primary model first

        last_error = None
        while True:
            current_model = fallback.get_current_model()
            if not current_model:
                raise last_error or RuntimeError("All models in fallback chain failed")

            # Use Logfire span for tracing
            with get_logfire_span(
                "llm_completion",
                model=current_model,
                agent=agent_name,
                temperature=temperature,
            ):
                try:
                    logger.info(
                        "LLM request",
                        model=current_model,
                        agent=agent_name,
                        temperature=temperature,
                    )

                    start_time = time.time()
                    response = completion(
                        model=current_model,
                        messages=msg_list,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=self.settings.llm.request_timeout,
                        **kwargs,
                    )
                    latency_ms = (time.time() - start_time) * 1000

                    # Extract response data
                    content = response.choices[0].message.content or ""
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }

                    # Calculate cost
                    cost = litellm.completion_cost(completion_response=response)
                    self._total_cost += cost

                    # Determine provider from model string
                    provider = current_model.split("/")[0] if "/" in current_model else "unknown"

                    # Log to Logfire
                    log_llm_call(
                        model=current_model,
                        tokens=usage["total_tokens"],
                        cost=cost,
                        latency_ms=latency_ms,
                        agent=agent_name,
                    )

                    logger.info(
                        "LLM response",
                        model=current_model,
                        tokens=usage["total_tokens"],
                        cost=cost,
                        latency_ms=latency_ms,
                    )

                    return LLMResponse(
                        content=content,
                        model=current_model,
                        provider=provider,
                        usage=usage,
                        cost=cost,
                        latency_ms=latency_ms,
                    )

                except (litellm.RateLimitError, litellm.APIError, litellm.Timeout) as e:
                    logger.warning(
                        "LLM request failed, trying fallback",
                        model=current_model,
                        error=str(e),
                    )
                    last_error = e
                    if not fallback.advance():
                        raise

                except Exception as e:
                    logger.error("LLM request failed", model=current_model, error=str(e))
                    raise

    async def acomplete(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_name: Optional[str] = None,
        use_fallback: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Async version of complete() with Logfire tracing.

        Args:
            Same as complete()

        Returns:
            LLMResponse with generated content
        """
        # Determine model and temperature
        if model is None and agent_name:
            model = self.get_model_for_agent(agent_name)
        model = model or self.settings.llm.default_model

        if temperature is None and agent_name:
            temperature = self.get_temperature_for_agent(agent_name)
        temperature = temperature if temperature is not None else self.settings.llm.default_temperature

        # Convert messages to dict format
        msg_list = [
            {"role": m.role, "content": m.content} if isinstance(m, LLMMessage) else m
            for m in messages
        ]

        with get_logfire_span(
            "llm_completion_async",
            model=model,
            agent=agent_name,
        ):
            try:
                start_time = time.time()
                response = await acompletion(
                    model=model,
                    messages=msg_list,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.settings.llm.request_timeout,
                    **kwargs,
                )
                latency_ms = (time.time() - start_time) * 1000

                content = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                cost = litellm.completion_cost(completion_response=response)
                self._total_cost += cost
                provider = model.split("/")[0] if "/" in model else "unknown"

                # Log to Logfire
                log_llm_call(
                    model=model,
                    tokens=usage["total_tokens"],
                    cost=cost,
                    latency_ms=latency_ms,
                    agent=agent_name,
                )

                return LLMResponse(
                    content=content,
                    model=model,
                    provider=provider,
                    usage=usage,
                    cost=cost,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error("Async LLM request failed", model=model, error=str(e))
                raise

    async def astream(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        agent_name: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream completion from LLM.

        Args:
            messages: List of messages
            model: Model to use
            temperature: Temperature setting
            agent_name: Agent name for config lookup
            **kwargs: Additional parameters

        Yields:
            Chunks of generated text
        """
        if model is None and agent_name:
            model = self.get_model_for_agent(agent_name)
        model = model or self.settings.llm.default_model

        if temperature is None and agent_name:
            temperature = self.get_temperature_for_agent(agent_name)
        temperature = temperature if temperature is not None else self.settings.llm.default_temperature

        msg_list = [
            {"role": m.role, "content": m.content} if isinstance(m, LLMMessage) else m
            for m in messages
        ]

        response = await acompletion(
            model=model,
            messages=msg_list,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def get_total_cost(self) -> float:
        """Get total cost incurred in this session."""
        return self._total_cost

    def reset_cost_tracking(self) -> None:
        """Reset cost tracking to zero."""
        self._total_cost = 0.0

    def validate_connection(self, model: Optional[str] = None) -> bool:
        """
        Validate connection to LLM provider.

        Args:
            model: Model to test (uses default if not provided)

        Returns:
            True if connection is valid
        """
        model = model or self.settings.llm.default_model
        try:
            response = self.complete(
                messages=[{"role": "user", "content": "test"}],
                model=model,
                max_tokens=5,
                use_fallback=False,
            )
            return bool(response.content)
        except Exception as e:
            logger.error("Connection validation failed", model=model, error=str(e))
            return False

    def list_available_models(self) -> List[str]:
        """List all configured models."""
        models = set()
        for agent_config in self.agent_mapping.values():
            if "model" in agent_config:
                models.add(agent_config["model"])
        for chain in self.fallback_chains.values():
            models.update(chain)
        return sorted(models)
