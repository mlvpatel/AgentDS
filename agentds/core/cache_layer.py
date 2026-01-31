"""
AgentDS Cache Layer.

Provides caching functionality with Redis and in-memory backends.

Author: Malav Patel
"""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import lru_cache
from typing import Any, Optional, TypeVar, Union

from agentds.core.config import Settings, get_settings
from agentds.core.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(
        self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all values from cache."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""

    def __init__(self) -> None:
        """Initialize memory cache."""
        self._cache: dict[str, tuple[Any, Optional[float]]] = {}
        self._max_size = 1000

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        import time

        if key not in self._cache:
            return None

        value, expiry = self._cache[key]
        if expiry and time.time() > expiry:
            del self._cache[key]
            return None

        return value

    def set(
        self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in memory cache."""
        import time

        # Evict if at max size
        if len(self._cache) >= self._max_size:
            # Remove oldest entries
            oldest_keys = list(self._cache.keys())[: self._max_size // 4]
            for k in oldest_keys:
                del self._cache[k]

        expiry = None
        if ttl:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            expiry = time.time() + ttl

        self._cache[key] = (value, expiry)
        return True

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.get(key) is not None

    def clear(self) -> bool:
        """Clear all values."""
        self._cache.clear()
        return True


class RedisCache(CacheBackend):
    """Redis cache implementation."""

    def __init__(self, url: str) -> None:
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
        """
        import redis

        self._client = redis.from_url(url, decode_responses=False)
        self._prefix = "agentds:"

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            data = self._client.get(self._make_key(key))
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.warning("Redis get failed", key=key, error=str(e))
            return None

    def set(
        self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in Redis."""
        try:
            data = pickle.dumps(value)
            if ttl:
                if isinstance(ttl, timedelta):
                    ttl = int(ttl.total_seconds())
                self._client.setex(self._make_key(key), ttl, data)
            else:
                self._client.set(self._make_key(key), data)
            return True
        except Exception as e:
            logger.warning("Redis set failed", key=key, error=str(e))
            return False

    def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            return bool(self._client.delete(self._make_key(key)))
        except Exception as e:
            logger.warning("Redis delete failed", key=key, error=str(e))
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self._client.exists(self._make_key(key)))
        except Exception as e:
            logger.warning("Redis exists failed", key=key, error=str(e))
            return False

    def clear(self) -> bool:
        """Clear all AgentDS keys from Redis."""
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor, match=f"{self._prefix}*", count=100
                )
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
            return True
        except Exception as e:
            logger.warning("Redis clear failed", error=str(e))
            return False

    def ping(self) -> bool:
        """Check Redis connection."""
        try:
            return self._client.ping()
        except Exception:
            return False


class CacheLayer:
    """
    Cache layer with automatic backend selection.

    Tries Redis first, falls back to memory cache if unavailable.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize cache layer.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._backend: CacheBackend
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize appropriate cache backend."""
        if self.settings.is_feature_enabled("redis_cache"):
            try:
                redis_cache = RedisCache(self.settings.redis.url)
                if redis_cache.ping():
                    self._backend = redis_cache
                    logger.info("Using Redis cache backend")
                    return
            except Exception as e:
                logger.warning("Redis unavailable, falling back to memory", error=str(e))

        self._backend = MemoryCache()
        logger.info("Using memory cache backend")

    def get(self, key: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        value = self._backend.get(key)
        return value if value is not None else default

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[Union[int, timedelta]] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds or timedelta

        Returns:
            True if successful
        """
        return self._backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self._backend.delete(key)

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self._backend.exists(key)

    def clear(self) -> bool:
        """Clear all cached values."""
        return self._backend.clear()

    def get_or_set(
        self,
        key: str,
        default_factory: Any,
        ttl: Optional[Union[int, timedelta]] = None,
    ) -> Any:
        """
        Get value from cache, or compute and cache if not present.

        Args:
            key: Cache key
            default_factory: Callable to produce default value
            ttl: Time to live

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        if callable(default_factory):
            value = default_factory()
        else:
            value = default_factory

        self.set(key, value, ttl)
        return value

    @staticmethod
    def make_key(*args: Any, **kwargs: Any) -> str:
        """
        Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Hash-based cache key
        """
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()


@lru_cache()
def get_cache() -> CacheLayer:
    """Get cached CacheLayer instance."""
    return CacheLayer()
