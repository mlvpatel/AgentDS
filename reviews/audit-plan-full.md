# Full Spectrum Audit: AgentDS Repository Health

**Date**: 2026-11-03
**Mode**: Repo Health. The branch matched `main` and the working tree was clean, so there was nothing to review for "uncommitted changes" or "branch vs main".
**Source**: `main@d29bbd5`. Line references in the lens reports point to that commit.
**Lenses Run**: 10/10. The lenses were written for a Tauri/Rust/React desktop app and were adapted to this Python stack:
- Frontend = Gradio presentation layer
- Backend = Python services (async, Redis, LangGraph, LiteLLM)
- Bundle/IPC = Docker, k8s, HTTP, Redis

**Lenses Skipped**: none.
**How it ran**:
- Design, UX, Frontend and Accessibility (small UI surface) were audited directly.
- Backend, Structural, Repo Maintainer, Production Readiness, DX and Performance each ran as an independent agent.
- Every critical item in the master tables was checked against the code. Items marked **[runtime-verified]** were reproduced in a scratch venv (`pip install -e ".[dev]"`, Python 3.13, gradio 6.31, langgraph 1.2).

**Baseline checks (main@d29bbd5)**:
- `ruff check .` passes.
- `pytest`: 169 passed, 15 deselected (the default `addopts` excludes LLM and integration tests). Coverage is 42% overall; `pipeline.py` 26%, `web/app.py` 10%, `cli.py` 0%.
- `mypy agentds/` reports 2 errors with current dependency versions (`middleware.py:276`, `pipeline.py:621`), so the CI type-check job fails on a fresh install.

> **Follow-up on this branch:** commit `d6703cd` fixes the pipeline routing and human-in-the-loop bugs (master items 1–4, plus partial fixes for items 5, 20 and 23). Each item's status is shown in the **Status** column below. All other items are still open.

---

## Master Verdict: NEEDS REWORK

The orchestration core shipped unable to finish in either mode. With human-in-the-loop on (the default everywhere), no agent ran. With it off, the first agent re-ran until LangGraph's recursion limit. Every interface (CLI, REST API, Gradio) still reported success. Three things kept this hidden: tests that never touch the pipeline, integration tests deselected by default, and type and lint gates switched off package-wide.

Outside the pipeline, the deployable surface also has release-blocking problems:
- The API's auth fails open.
- Redis is published without a password and its contents are unpickled.
- A paid, blocking LLM call serves as the liveness probe.
- The web container crashes at startup when no Logfire token is set.
- The documented quick start and k8s rollout both break.

## Verdict by Lens

| # | Lens | Verdict | Critical | Recommendations |
|---|------|---------|----------|-----------------|
| 1 | Design Engineer | APPROVE WITH CHANGES | 0 | 5 |
| 2 | UX Engineer | NEEDS REWORK | 5 | 7 |
| 3 | Frontend Engineer (Gradio) | NEEDS REWORK | 3 | 8 |
| 4 | Backend Engineer | NEEDS REWORK | 10 | 12 |
| 5 | Structural Engineer | NEEDS REWORK | 7 | 12 |
| 6 | Repo Maintainer | NEEDS REWORK | 5 | 12 |
| 7 | Production Readiness | NEEDS REWORK | 10 | 12 |
| 8 | DX Engineer | NEEDS REWORK | 9 | 12 |
| 9 | Performance Engineer | NEEDS REWORK | 5 | 12 |
| 10 | Accessibility Engineer | APPROVE WITH CHANGES | 0 | 5 |

## All Critical Issues (Must Fix)

Deduplicated across lenses and ordered by severity.

| # | Issue | Lens(es) | File(s) | Impact | Fix | Status |
|---|-------|----------|---------|--------|-----|--------|
| 1 | Moving to the next agent happens inside a LangGraph conditional-edge router, which discards state changes **[runtime-verified: DataLoaderAgent ran 10,006×, then GraphRecursionError]** | Backend C1, Structural C3, UX, Frontend | `workflows/pipeline.py:440-444` | Every `--no-hitl` or `human_in_loop=false` run loops on the first agent (paying for LLM calls) and never finishes | Advance in a node; keep routers pure | **Fixed** (`d6703cd`) |
| 2 | HITL mode (the default) never invokes the graph. There is no `interrupt_before`, and `resume()` restarts from preflight. The UI review controls are hidden and `job_id_state` is never set **[runtime-verified: 0 agents ran, job left QUEUED]** | Backend C2, Structural C2, Prod C5, Repo C3, DX C2, UX U2/U3, Frontend F1 | `pipeline.py:559-567, 604-624`; `app.py:67, 151, 328-353` | The headline feature does nothing | Compile with `interrupt_before=["human_review"]`, always invoke, resume via `update_state(as_node=…)` | **Fixed** (`d6703cd`) |
| 3 | False success. CLI, UI and API report success regardless of outcome, and failed agents are recorded as COMPLETED | UX U1, Backend C8, DX C2, Prod C8 | `cli.py:111`, `app.py:322`, `webhooks.py:187-192`, `pipeline.py:326-337, 570-575` | Users and scripts act on results that don't exist; CLI exits 0 | A status contract from `run`/`resume`; failed agent → error, or review in HITL mode | **Fixed** (`d6703cd`) |
| 4 | The job ID returned to callers never matches the stored Job, and each entry point builds its own JobQueue **[runtime-verified]** | Backend C3, Structural C1, Prod C5c, Repo C4, DX C6, UX C4 | `pipeline.py:144, 522-534`; `webhooks.py:121, 211`; `job_queue.py:48, 322` | `/pipeline/status/{id}` and cancel return 404; the n8n polling flow never completes | `create_job(job_id=…)`; inject a shared queue | **Fixed** (`d6703cd`); `rq` still undeclared (see #24) |
| 5 | A polars DataFrame in agent outputs crashes checkpointing and JSON responses **[runtime-verified: `TypeError: Type is not msgpack serializable: DataFrame`]** | Structural C5, Perf R1 | `data_cleaning.py:168` → `pipeline.py:331` | Exposed as soon as #1 and #2 are fixed | Only JSON-safe values in outputs | **Mitigated**: pipeline sanitizes outputs at the state boundary; the agent still returns the frame |
| 6 | API auth fails open with no keys; `data_source`/`output_destination` reach the filesystem, `httpx.get` and DB DSNs unvalidated; `validation.py` is never called | Backend C6, Structural C7, Prod C2, Repo R6 | `middleware.py:217-220`; `webhooks.py:174-179, 335-344`; `data_loader.py:240-330` | Anonymous LLM spend, local file read, SSRF | Fail closed; validate at the web boundary | Open |
| 7 | Redis published on the host without auth, and job records read back with `pickle.loads` | Backend C7, Prod C9 | `docker-compose.yml:88`; `job_queue.py:226, 252` | Remote code execution in containers that hold the LLM keys | Store jobs as JSON; don't publish 6379; `requirepass` | Open |
| 8 | Web/CLI crash at startup when no `LOGFIRE_TOKEN` is set **[runtime-verified: `LogfireConfigError`]** | Prod C1, DX C4 | `core/logger.py:44-61` | Default deployment and `agentds status` fail; CI hides it via `ENVIRONMENT=test` | `send_to_logfire="if-token-present"`; catch `Exception` | Open |
| 9 | `cp .env.example .env` breaks `Settings()` **[runtime-verified: `SettingsError` on `api_keys`]** | DX C3, Repo C2, Prod C2 | `.env.example:134`; `config.py:241, 261-267` | Quick-start step 2 breaks every entry point | `Annotated[list[str], NoDecode]`, or comment the line out | Open |
| 10 | `/api/health` makes a blocking, retried, paid LLM call and serves as liveness and readiness | Backend C5, Perf C3, Prod C3, Structural R2 | `webhooks.py:137-147`; `k8s/deployment.yaml:259-270` | Restart loops whenever LLM latency >1s; ~16 billed calls/min | Process-only liveness; Redis-ping readiness; cached LLM check | Open |
| 11 | Pipelines and agents run synchronously inside `async` handlers; the RQ worker has nothing to consume | Backend C4, Perf C2, Prod C4, Structural R1 | `webhooks.py:174, 285, 348`; `job_queue.py:204` | One run freezes the API, probes fail, and the pod is killed mid-run | Enqueue to the worker and return 202 + job_id | Open |
| 12 | The LLM fallback chain mutates the shared config list on every call **[runtime-verified: 5 → 10 entries after 5 calls]** | Backend C10, Perf C1, Structural C6, Prod R1 | `llm_gateway.py:178-181, 230` | Outages turn into long retry storms against already-failing models | Copy the list; dedupe; cap attempts | Open |
| 13 | The APIWrapperAgent template is an f-string containing `{field: … for field in …}`, which raises on every call **[runtime-verified f-string behaviour]** | Repo C1 | `api_wrapper.py:343, 382`; hidden by `pyproject.toml:143` F821 ignore | Deploy phase can never succeed | Escape as `{{…}}`; remove the F821 ignore; compile-test the template | Open |
| 14 | The generated `app.py` pastes raw column names into Python source | Backend C9 | `api_wrapper.py:216-218, 343-352` | SyntaxError in the deployed service; code injection through CSV headers | Sanitize identifiers; `Field(alias=…)`; `compile()` before saving | Open |
| 15 | The EDA → FeatureEngineer handoff expects `data_path`, which EDA never outputs | Structural C4 | `feature_engineer.py:109-119`; `eda_copilot.py:142-148` | Every BUILD run with EDA now stops at FeatureEngineerAgent (made visible by #1–#3) | Resolve input from the latest completed result that has `data_path`, or use typed outputs | Open |
| 16 | k8s rollout: the Secret is overwritten with placeholders, the UID doesn't match (999 vs 1000), the RWX PVC is unsupported, and the image is never published | Prod C6, C7; Repo C5 | `k8s/deployment.yaml:35-43, 145, 195-197, 233`; `Dockerfile:37`; `release.yml` | The documented deploy is down or every LLM call returns 401 | Remove the Secret from the manifests; `useradd -u 1000` + `fsGroup`; RWO; add an image publish job | Open |
| 17 | Agent models need provider keys that the deploy configs never inject (e.g. Groq); `google/` is not a LiteLLM prefix | Prod C8, DX R11 | `config/llm_config.yaml:34-47`; `k8s/deployment.yaml:155-164` | The first agent fails on every k8s run | Validate mapped models and keys at startup | Open |
| 18 | Outputs and run state from API-triggered pipelines are lost on pod restart | Prod C10 | `k8s/deployment.yaml:215-270`; `pipeline.py:152` | Models and reports vanish; paused runs are orphaned | Shared storage for outputs and temp files; persistent checkpointer | Open |
| 19 | Dense one-hot encoding with no cardinality guard | Perf C5 | `feature_engineer.py:227-228, 304-308` | Out-of-memory on any ID-like string column | Cardinality cap; `sparse_output=True` | Open |
| 20 | `_active_pipelines` registries grow without bound and live per process | Perf C4, Frontend F2, Backend R4 | `app.py:26`; `webhooks.py:112` | Memory leak; HITL breaks across replicas | TTL or persistent checkpointer | **Partly fixed**: entries are now held only while awaiting review and removed when the run ends; still per process |
| 21 | Quick-start docs fail: the README snippet calls `run(task=…)`, the `PipelineConfig` import fails, 8 of 10 agents can't run alone via `agentds agent`, pre-commit isn't configured, and there is no `develop` branch | DX C1, C7, C8, C9; Repo R3, R4, R11 | `README.md:129-135`; `agentds/__init__.py`; `cli.py:128-167`; `CONTRIBUTING.md` | First-run failure on every documented path | Fix the docs and exports; add a docs-snippet smoke test | Open |
| 22 | The Configuration tab collects API keys and silently discards them | UX U4, Prod N7 | `app.py:196-232` | Users think keys are set; secrets typed into a no-op field | Remove the tab, or wire it with clear "this session" feedback | Open |
| 23 | Fake progress: the UI blocked for the whole run, then replayed a `time.sleep(0.5)` animation | UX U5, Frontend F3, Perf R7 | `app.py:297-323` | No real feedback during long runs | Stream LangGraph node events from a worker | **Partly fixed**: the fake loop is removed and the UI shows "Running…" plus the real result; still no streaming |
| 24 | `rq` is not a declared dependency, so the Redis queue silently falls back to memory **[runtime-verified: `import rq` fails after `pip install -e .[dev]`]** | Structural C1, Repo R5, UX | `pyproject.toml`; `job_queue.py:200-201, 293-298` | Jobs never persist or show up across processes | Declare `rq` (or drop RQ) and fail loudly | Open |

## Cross-Cutting Concerns

| Concern | Flagged By | Root Cause | Unified Fix |
|---------|-----------|------------|-------------|
| Success reported regardless of outcome | UX, Frontend, Backend, Structural, Prod, Repo, DX | `run()` returned an untyped dict; callers treated "no exception" as success; agents swallow their own errors | A status contract on `run`/`resume` (done in `d6703cd`). Next: typed `PipelineResult` with `.ok` and `raise_for_error()` |
| No execution boundary | Backend, Perf, Prod, Structural, Frontend | Web handlers call the pipeline directly; RQ is wired but unused | A `PipelineService` that enqueues to the worker; handlers only enqueue and read status |
| State that is process-local and not durable | Backend, Structural, Prod, Perf, Frontend | `_active_pipelines`, `MemorySaver`, `InMemoryQueue` and per-pod rate limits | Redis or Postgres checkpointer keyed by job_id, JSON job store, rebuild the pipeline on resume |
| Untrusted input reaches the filesystem, network and SQL | Backend, Structural, Prod, Repo | `validation.py` orphaned; auth fails open; LLM-supplied SQL executed | Validate at the web boundary; fail-closed auth; never execute LLM SQL |
| Health and readiness coupled to a paid LLM call | Backend, Perf, Prod, Structural | `/api/health` constructs a gateway and calls `validate_connection()` | Split live/ready; cached background LLM check |
| Shared mutable LLM gateway state | Backend, Perf, Structural, Prod | Fallback list mutation, `os.environ` writes, cumulative cost counter | Copy-on-read config; per-call keys; per-agent usage deltas |
| Redis job store design | Backend, Perf, Prod, Structural, Repo | One pickled hash under `allkeys-lru`, no timeouts, `rq` undeclared | One JSON key per job with a ZSET index, timeouts, `noeviction` |
| Docs and config that don't match code | DX, Repo, Prod, UX | Unread `pipeline_config.yaml`, presets, flags and env vars; wrong snippets | One source of truth per setting; snippet smoke tests in CI |
| Quality gates that cannot fail | Repo, Frontend, Backend, Structural | 13 mypy codes disabled, an F821 ignore masking a real NameError, `\|\| true` on security steps, integration tests deselected, API tests accepting 500 | Re-enable gates package by package; assert exact codes; graph-level tests (19 added in `d6703cd`) |

## All Recommendations

The highest-value items, deduplicated. Each lens report below has its full list.

| # | Recommendation | Lens | File(s) | Fix |
|---|---------------|------|---------|-----|
| 1 | Make CI gates real | Repo | `.github/workflows/ci.yml`, `pyproject.toml` | Drop `\|\| true`, add `ruff format --check` and `--cov-fail-under`, re-enable mypy codes incrementally, pin tool versions |
| 2 | Single dependency source; declare `click` and `rq`; prune unused deps (duckdb, llama-index, pydantic-ai, mlflow) | Repo, Structural, Perf | `pyproject.toml`, `requirements.txt`, `Dockerfile` | Docker runs `pip install .` against a lock file |
| 3 | Atomic, concurrency-safe artifact index | Backend, Prod, Perf | `artifact_store.py:81-110` | Temp file + `os.replace`, per-job metadata or Redis |
| 4 | Redis clients with timeouts and one shared pool | Perf, Prod, Backend | `job_queue.py:203`, `cache_layer.py:131` | `ConnectionPool.from_url(..., socket_timeout, max_connections)` |
| 5 | Widen the LiteLLM retryable and fallback exception set; guard `completion_cost` | Backend, Prod | `llm_gateway.py:183-187, 273, 305-317` | Classify 5xx, connection and auth errors |
| 6 | Validate LLM JSON with `call_llm_structured` instead of six hand-rolled regex parsers | Backend, Structural, Repo | 6 agents; `base.py:223` | Pydantic response models with deterministic defaults |
| 7 | Clean temp files and cap ingestion size | Backend, Perf, Prod | `data_loader.py`, `feature_engineer.py`, `automl.py` | Per-job `TemporaryDirectory`; streaming with size caps |
| 8 | Bound AutoML compute | Perf | `automl.py:296-333` | Clamp trials and folds, add timeout and pruner, `n_jobs` = CPU limit |
| 9 | Middleware hardening | Backend, Prod | `middleware.py`, `webhooks.py:498-512` | Rate-limit before auth, headers-only keys, `hmac.compare_digest`, explicit CORS origins |
| 10 | Lazy package exports; no directory creation on `Settings()` | Structural, DX, Perf, Repo | `agentds/__init__.py`, `config.py:269-279` | PEP 562 `__getattr__`; create directories on write |
| 11 | Single source for agent order and identity | Structural | `pipeline.py`, `agents/__init__.py`, `app.py`, `cli.py`, `optimization.py` | Derive from class attributes or `pipeline_config.yaml` |
| 12 | Gradio: move theme and css to `launch()`, cap `gradio<7`, default bind to 127.0.0.1, support `auth=` | Design, Frontend | `app.py:62-66, 422-451` | Version-safe, private by default |
| 13 | Logs to stderr plus a real file handler; text format on a TTY | DX, Prod, UX | `logger.py`, `cli.py:35`, `app.py:438` | Makes the Logs tab and documented log path work |
| 14 | CLI argument validation (`click.Choice` for phases and agents; accept URLs) and honour `-o` | UX, DX | `cli.py` | Usage errors instead of tracebacks |
| 15 | Fix the DevOps feature-flag lookup (nested keys) and file generation | Structural, Backend, Repo | `config.py:296-311`, `devops.py` | Dotted lookup; strip code fences; create parent directories |
| 16 | SECURITY.md reporting channel; fix CODEOWNERS handle and ruleset check names | Repo | `SECURITY.md:17`, `.github/` | Private vulnerability reporting enabled |
| 17 | Accessibility: real progress element, announced status updates, heading hierarchy | A11y | `app.py:72-145` | `gr.Progress` / `role="status"`, `##` section headings |
| 18 | Move `REVIEW_FEEDBACK.md` / `REVIEW_SUMMARY.md` out of the root (several claims are now false) | Repo | repo root | Turn the findings into issues, or move the files to `docs/reviews/` |

---

# Lens Reports

Lens reports 4–9 are reproduced as their auditing agents returned them, with headings demoted to fit this document. Their findings were spot-checked; the critical items are cross-referenced in the master tables above.

---

## 1. Design Engineer

**Verdict**: APPROVE WITH CHANGES

**Perspective**: Design Engineer · **Date**: 2026-11-03 · **Source**: Repo Health — main@d29bbd5

### Summary
The only visual surface is the Gradio Blocks UI in `agentds/web/app.py`, which uses `gr.themes.Soft()` plus a small inline CSS block. Gradio supplies most of the look, so there are few design problems. The problems that do exist are dead or hardcoded CSS that falls outside Gradio's theme tokens, a missing visual language for status, and a Gradio 6 deprecation on how the theme and CSS are passed in.

### Files Reviewed
| File | Visual Role | Risk |
|------|-------------|------|
| `agentds/web/app.py:42-78` | Theme, custom CSS, header | Medium |
| `agentds/web/app.py:80-253` | Tab layouts, cards, status areas | Medium |

### Critical Issues (Must Fix)
None. No design issue here blocks the user on its own; the blocking UI problems are behavioural and are listed under UX and Frontend.

### Recommendations
| # | Issue | File(s) | Why | Fix |
|---|-------|---------|-----|-----|
| D1 | `theme=` and `css=` are passed to the `gr.Blocks(...)` constructor. With the Gradio version that actually installs (6.31.0, since `pyproject.toml` has `gradio>=5.49.0` with no upper bound), this raises `UserWarning: The parameters have been moved from the Blocks constructor to the launch() method in Gradio 6.0: theme, css` (reproduced in the venv) | `app.py:62-66`, `pyproject.toml` | The Soft theme and custom CSS may be dropped on future Gradio releases, and the UI's look depends on an unpinned major version | Pass `theme`/`css` to `app.launch()` in `launch_app` (`app.py:447`), and pin `gradio>=5.49,<7` until the move is tested |
| D2 | 5 of the 6 custom CSS classes are never applied: `.status-running/.status-completed/.status-failed/.status-paused` and `.progress-bar`. Only `.agent-card` is used (`app.py:186`) | `app.py:43-58` | Dead design vocabulary suggests status styling that doesn't exist | Delete them, or apply them to the status Markdown via `elem_classes` (see D4) |
| D3 | Hardcoded hex colours (`#3b82f6`, `#22c55e`, `#ef4444`, `#f59e0b`, `#e5e7eb`, `#8b5cf6`) and magic numbers (`8px` radius, `16px` padding) instead of Gradio theme CSS variables | `app.py:43-58` | They bypass the Soft theme, and the light-grey `.agent-card` border (`#e5e7eb`) does not adapt to Gradio dark mode | Use `var(--border-color-primary)`, `var(--radius-lg)`, `var(--spacing-lg)`, `var(--color-accent)` and similar |
| D4 | Status, errors and success all render as plain Markdown in one style (`status_md`, `app.py:133`). `"Error: …"` (`:268, :326`) looks the same as `"Running: …"` (`:311`) | `app.py:133, 268-326` | Users can't tell failure from progress at a glance | Show errors with `gr.Warning`/`gr.Error` toasts, or style status with the (fixed) status classes plus an icon or word, never colour alone |
| D5 | A non-interactive `gr.Slider` is used as the progress bar | `app.py:136-142` | It looks like a disabled input control rather than progress | Use Gradio's built-in `gr.Progress()` tracker in the generator, or an HTML `<progress>` element |

### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| D6 | The Agents tab shows only bold names in bordered rows (`app.py:178-187`) under a heading that says "Agent Status" | Add per-agent state badges (idle/running/done/failed) once real status is wired in; otherwise rename it "Agents" |
| D7 | The Jobs Dataframe has no empty-state copy (`app.py:239-242`) | A short "No jobs yet — start a pipeline in the Pipeline tab" reads better than an empty grid |

### Design System Compliance
The UI correctly delegates to a Gradio theme (`gr.themes.Soft()`), which supplies tokens for colour, spacing and radius in light and dark modes. The custom CSS works against that system: it brings in Tailwind-palette hex values and fixed pixel sizes that ignore the theme variables, and most of the classes are dead. The fragmentation is small because so little custom CSS is in use.

### Visual State Coverage
| Component/Element | Hover | Focus | Active | Disabled | Loading | Empty | Error | Dark Mode |
|-------------------|-------|-------|--------|----------|---------|-------|-------|-----------|
| Start Pipeline button | Gradio | Gradio | Gradio | Not disabled during a run | Gradio's generic pending overlay only | N/A | Plain text | Gradio |
| Status Markdown | N/A | N/A | N/A | N/A | No stage info | "Ready to start..." | Same style as success | Gradio |
| Progress slider | N/A | Focusable as a slider | N/A | Shown as disabled | Fake values played back after completion | 0 | Reset to 0 | Gradio |
| HITL buttons | — | — | — | Permanently hidden (`visible=False`) | — | — | — | — |
| Agents cards | none | none | none | N/A | none | N/A | N/A | Light-only border |
| Jobs table | Gradio | Gradio | N/A | N/A | none | No copy | Raises on backend error | Gradio |
| Logs code view | N/A | Gradio | N/A | N/A | none | "No logs available" (always; see UX) | none | Gradio |

### Verdict Details
- Design Token Fidelity: CONCERNS (hex values and magic numbers bypass the theme)
- Visual State Coverage: CONCERNS (no error or loading styling)
- Dark Mode Parity: CONCERNS (`.agent-card` border)
- Animation Quality: PASS (no custom motion)
- Visual Consistency: CONCERNS (dead status vocabulary; slider used as progress)

---

## 2. UX Engineer

**Verdict**: NEEDS REWORK

**Perspective**: UX Engineer · **Date**: 2026-11-03 · **Source**: Repo Health — main@d29bbd5

### Summary
Users reach AgentDS through the Gradio UI (`agentds web`) and the CLI (`agentds run|agent|jobs|status|config`). On both, the main promise of the product (run the pipeline and review each agent) is not delivered, yet both report success. I confirmed this by running `AgentDSPipeline.run()` in a venv with agents stubbed. With `human_in_loop=True`, the default in the UI checkbox, the CLI and `PipelineConfig`, zero agents ran, the job stayed `QUEUED`, and the call returned normally. With `human_in_loop=False`, the first agent ran 10,006 times before `GraphRecursionError`.

### User Journey Map — Web UI
| Step | Trigger → Action → Feedback → Outcome | Mark |
|------|----------------------------------------|------|
| 1 | `agentds web` → prints `Launching … on 0.0.0.0:7860` (`cli.py:47`) | OK (the 0.0.0.0 URL isn't clickable, but Gradio prints the local URL) |
| 2 | Pipeline tab → upload a file or enter a URL, describe the task | OK |
| 3 | Advanced Options → phases `build/deploy/learn` with no explanation; HITL checkbox on by default; **LLM Preset dropdown has no effect** (`llm_preset` is accepted at `app.py:263` and never used) | WARN / FAIL |
| 4 | Click **Start Pipeline** → `pipeline.run()` runs synchronously before the first `yield` (`app.py:298`), so the only feedback is Gradio's generic pending overlay | FAIL |
| 5 | Progress → after `run()` returns, the UI loops over `pipeline._agents` with `time.sleep(0.5)` and "Processing..." (`app.py:308-319`). The progress is fake and is played back afterwards | FAIL |
| 6 | Outcome → always `"Pipeline completed successfully!"` (`app.py:322`), even when no agent ran (HITL on) or `state["error"]` is set (`run()` returns normally with the error inside `result["state"]`, `pipeline.py:570-581`) | FAIL |
| 7 | Human review → the Approve, Re-run, Skip and Stop row is `visible=False` and nothing ever shows it (`app.py:151`). The feedback box and the "Re-run with Feedback" button are also hidden and not wired (`:158-166`) | FAIL |
| 8 | Artifacts → raw result JSON; the **Download All Outputs** button is not wired (`app.py:171`) | FAIL |
| 9 | Configuration tab → the user types API keys, picks a model and clicks **Save Configuration**. Nothing is wired (`app.py:196-232`), there's no feedback, and the keys are silently discarded | FAIL |
| 10 | Jobs tab → **Refresh** lists `job_queue` from `create_app` (`app.py:40`), but each pipeline creates its own `JobQueue` (`pipeline.py:145`). On the in-memory backend, which is what you get from a `pip install` because `rq` isn't a declared dependency, pipeline jobs never appear | FAIL |
| 11 | Logs tab → reads `logs/agentds.log` (`app.py:371`), but nothing configures a file handler (`setup_logging` is never given `log_file`; grep finds no `log_file=` call site), so it always shows "No logs available" | FAIL |

### User Journey Map — CLI
| Step | Trigger → Action → Feedback → Outcome | Mark |
|------|----------------------------------------|------|
| C1 | `agentds run data.csv -t "…"` (HITL on by default) → returns at once and prints `[OK] Pipeline completed successfully!` (`cli.py:111`) with nothing executed | FAIL |
| C2 | `agentds run … --no-hitl` → the first agent loops until `GraphRecursionError`. Any other failure that ends up in `result["state"]["error"]` still prints `[OK]` | FAIL |
| C3 | `--phases foo` → `PipelinePhase("foo")` raises a raw `ValueError` traceback outside the `try` (`cli.py:95`) | WARN |
| C4 | `run` prints `Job ID: {result['job_id']}` (`cli.py:112`), but the stored job has a different UUID (`Job.id` default factory, `job_queue.py:48`; `create_job` ignores the pipeline's `job_id`, `pipeline.py:526`), so `agentds jobs` can't be matched to it | FAIL |
| C5 | `agentds agent X data.csv -o out` → `--output` is accepted and never used (`cli.py:123-125`) | WARN |
| C6 | `agentds jobs` → a new process with the in-memory backend shows `No jobs found` | WARN |
| C7 | `agentds status` → reports Redis `[OK]` if it pings (`cli.py:207-209`), even though the queue can't use Redis without `rq` | WARN |
| C8 | `run` accepts only existing local paths (`click.Path(exists=True)`, `cli.py:67`), while the UI and docs accept `s3://` and `https://` | WARN |

### Critical Issues (Must Fix)
| # | Issue | Step in Journey | Impact on User | Fix |
|---|-------|----------------|----------------|-----|
| U1 | **False success.** The UI and CLI report success whatever actually happened (`app.py:322`, `cli.py:111`); neither checks `result["state"]["completed"]` or `["error"]` | 6, C1, C2 | Users think models and deploy artifacts were built when nothing ran, and they lose trust after the first real check | Inspect `state.completed`/`state.error`/`job.status`; show a distinct failure state and exit non-zero in the CLI |
| U2 | **The default mode does nothing.** HITL on (the default everywhere) makes `run()` skip graph execution (`pipeline.py:560-567`) | 4-7, C1 | The product's main flow is a no-op unless users find `--no-hitl`, which then loops | Fix the pipeline (see the Backend/master issue). Until HITL actually pauses and resumes, default `human_in_loop=False` or refuse to start with a clear message |
| U3 | **The HITL controls can't be reached.** The review row is permanently hidden (`app.py:151`) and `job_id_state` is never written (`app.py:67`, missing from `start_pipeline` outputs `:382`), so every handler returns "No active pipeline" | 7 | "Human-in-the-loop", the headline feature, can't be used in the UI | Return `job_id` into `job_id_state`; toggle row visibility with `gr.update(visible=...)` when `awaiting_approval`; call `pipeline.resume(job_id, action, feedback)` from the handlers |
| U4 | **The Configuration tab collects secrets and drops them.** Six controls and a Save button are unwired (`app.py:196-232`) | 9 | Users believe their keys are set, then hit `Pre-flight checks failed: LLM connection failed`. They've also pasted secrets into a field that does nothing | Either remove the tab and point users to `.env`/`agentds config`, or wire Save to an in-session settings override with clear "saved for this session" feedback |
| U5 | **Fake progress.** The progress animation plays after the pipeline has finished (`app.py:304-319`) | 4-5 | During the real run users get no stage info and can't tell a 20-minute AutoML run from a hang | Run the pipeline in a worker and stream real node events (LangGraph `.stream()` / `stream_mode="updates"`), yielding on each node transition |

### Recommendations
| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| U6 | The LLM Preset dropdown is ignored (`app.py:115-119, 263`) | A control with no effect | Wire it to `LLMGateway` presets or remove it |
| U7 | The Download All Outputs button isn't wired (`app.py:171`) | Dead end | Zip `output_destination` and return it via `gr.File`, or remove the button |
| U8 | The Logs tab can never show logs (`app.py:369-376`) | Dead tab | Pass `log_file=settings.log_dir / "agentds.log"` in `launch_app`/CLI setup, and tail it with `collections.deque(f, maxlen=100)` |
| U9 | Job ID mismatch between `run()`'s returned `job_id` and the stored `Job.id` | Users can't trace a run | Let `create_job` accept an `id` and pass `job_id` through |
| U10 | The CLI `--phases` option isn't validated | Traceback on a typo | `type=click.Choice([p.value for p in PipelinePhase])` |
| U11 | `agentds agent -o` is ignored | Outputs land somewhere other than where the user asked | Put `output` into `context.extra["output_destination"]` |
| U12 | Raw exception text is shown to users (`app.py:326`, `cli.py:116`) | Errors aren't actionable | Map `AgentDSError` subclasses to messages that say what to do; show the traceback only with `--debug` (`ctx.obj["debug"]` is set at `cli.py:31` but never read) |

### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| U13 | Add one-line explanations for the build/deploy/learn phases in the UI and CLI help | Users new to the tool don't know what "learn" does |
| U14 | Accept URLs in `agentds run` (drop `exists=True`; validate in preflight) | Matches the UI and docs |
| U15 | Disable the Start button while a run is in flight (`gr.update(interactive=False)`) | Prevents double-submits that each start a full pipeline |

### Interaction Consistency Check
The UI and CLI disagree on what a data source can be (URL versus local path only) and on how the job ID is shown (UI: never; CLI: an ID nobody can look up). Both share the same failure: they assume `run()` succeeded if it didn't raise. The UI offers controls (preset, save configuration, download, review buttons) with no behaviour behind them, which teaches users that the interface can't be trusted.

### Verdict Details
- Discoverability: CONCERNS (HITL controls hidden; logs and jobs tabs empty)
- Flow Completeness: CONCERNS (the core run is a no-op or loops; dead ends at review, config and download)
- Error Recovery: CONCERNS (false success; raw errors; no resume)
- Feedback & Status: CONCERNS (fake progress; no real stage feedback)
- Consistency: CONCERNS (UI and CLI data-source and job-ID mismatches)

---

## 3. Frontend Engineer

**Verdict**: NEEDS REWORK

**Perspective**: Frontend Engineer (applied to the Gradio presentation layer; there is no React or TypeScript) · **Date**: 2026-11-03 · **Source**: Repo Health — main@d29bbd5

### Summary
The frontend is a Gradio `Blocks` app built in Python. This lens covers component wiring, session state, the event and generator "render path", type flow into the components, and import weight. Test coverage of `agentds/web/app.py` is 10% (`pytest --cov`, run in the venv).

### Files Reviewed
| File | Frontend Role | Risk |
|------|--------------|------|
| `agentds/web/app.py` | All UI components, state and event handlers | High |
| `agentds/workflows/pipeline.py:503-629` | `run`/`resume` API consumed by the UI | High |
| `agentds/core/job_queue.py:266-349` | Jobs tab data source | Medium |

### Critical Issues (Must Fix)
| # | Issue | File(s) | Why It Breaks | Code Fix |
|---|-------|---------|---------------|----------|
| F1 | `job_id_state` is never written. `start_pipeline` outputs are `[status_md, progress_bar, agent_output, artifacts_list]` (`app.py:382`), so `job_id_state` stays `None` and every review handler takes the `"No active pipeline"` branch (`app.py:330-353`) | `app.py:67, 257-353, 379-407` | The HITL interaction is dead even once the row is made visible | Add `job_id_state` (and the review-row `gr.Row`) to the outputs; `yield …, job_id, gr.update(visible=awaiting)` |
| F2 | Module-level `_active_pipelines` (`app.py:26`) is shared across **all browser sessions and users**, keyed only by UUID, and never pruned. Each entry holds a full `AgentDSPipeline` (LLM gateway, artifact store, job queue, 10 agent instances, compiled graph) | `app.py:26, 290-294, 309` | Unbounded memory growth for as long as the process runs, and cross-user state in a multi-user deployment | Keep per-session data in `gr.State`; if a registry is needed, bound it (TTL/LRU) and delete on completion or stop |
| F3 | Blocking work happens before the first yield, and progress is synthetic. `pipeline.run()` blocks inside the generator (`app.py:298`); progress is then simulated with `time.sleep(0.5)` per agent (`:318`). With Gradio's default `concurrency_limit=1` per event, one user's run blocks all other users' Start clicks | `app.py:297-323` | The UI is unresponsive for the whole run, other users are serialized behind it, and the progress shown is fiction | Stream real updates with `compiled.stream(..., stream_mode="updates")` and yield on each node; set an explicit `concurrency_limit` on `start_btn.click` |

### Recommendations
| # | Issue | File(s) | Why | Fix |
|---|-------|---------|-----|-----|
| F4 | Unwired components: the API key textboxes, model dropdown, temperature slider and Save button (`:196-232`); "Re-run with Feedback" (`:163-166`); "Download All Outputs" (`:171`); `llm_preset` input (`:263`) | `app.py` | Dead controls, and the handles aren't even assigned to variables | Wire them or delete them |
| F5 | The UI reaches into the private `pipeline._agents` (`app.py:308`) | `app.py:308`, `pipeline.py:147` | Couples the UI to pipeline internals | Expose a public `agents_for(phases)` or rely on streamed events |
| F6 | An unused `gr.State(value={})` is created (`app.py:68`) | `app.py:68` | Leftover state | Remove it |
| F7 | `theme`/`css` are passed to the `Blocks()` constructor, which is deprecated in the resolved Gradio 6.31 (reproduced warning); Gradio isn't capped at a major version | `app.py:62-66`, `pyproject.toml` | The next major version may break the UI | Move them to `launch()`; cap `gradio<7` |
| F8 | Handler results aren't type-checked against the components: `start_pipeline` yields the whole `result` dict (including `state` and `job.model_dump()` values) into `gr.JSON` (`app.py:323`) | `app.py:264, 323` | Large, internal and possibly non-JSON-safe values reach the client | Send a curated summary: `{"job_id", "status", "final_outputs"}` |
| F9 | `refresh_logs` reads the entire log file into memory to keep 100 lines (`f.readlines()[-100:]`, `app.py:374`) | `app.py:369-376` | Grows with the log size | `deque(f, maxlen=100)` |
| F10 | `refresh_jobs` doesn't handle Redis errors (`app.py:357`), and `RedisQueue.list_jobs` unpickles every job on each refresh | `app.py:355-367`, `job_queue.py:251-256` | One bad entry or a Redis outage turns into a generic Gradio error toast | Wrap it in try/except with a readable message; paginate |
| F11 | Launching the Gradio UI with no auth on `0.0.0.0` by default (`app.py:423`, `cli.py:39`), with an optional `share=True` public tunnel | `app.py:422-451` | A form that accepts API keys and arbitrary data URLs is exposed on every interface | Default to `127.0.0.1`; support `auth=` from settings; warn when `share` is used |

### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| F12 | Build the static Agents tab from `AgentDSPipeline.PHASE_AGENTS` instead of a hand-copied list (`app.py:178-182`) | Avoids drift from the real agent list |
| F13 | Add a smoke test that calls `create_app()` and asserts the components are wired | Coverage of the 464-line UI is 10%; the Gradio 6 deprecation would have been caught |

### Render Path Analysis
`start_btn.click` → the `start_pipeline` generator. (1) Validation yields early on missing input; that part is correct. (2) It constructs a new `AgentDSPipeline`, which instantiates 10 agents, an LLM gateway, an artifact store and a fresh `JobQueue`, and stores it in a global dict. (3) It calls `pipeline.run()` **synchronously**. With HITL on this returns immediately without executing; with HITL off it blocks until `GraphRecursionError` (confirmed by probe). (4) Only then does it yield N times, one per agent, with fake values and a 0.5 s sleep. (5) A final yield claims success. Because steps 2-3 run before any yield, Gradio shows only its generic pending overlay. Because the default per-event concurrency is 1, other sessions' Start clicks queue behind this generator. The review handlers render only into `status_md` and `feedback_input`; none of them change visibility, so the UI tree never changes after the initial build.

### Type Safety Assessment
`mypy agentds/` reports 2 errors on fresh deps (`middleware.py:276`, `pipeline.py:621`). `app.py` has none, but mostly because the pyproject override disables `arg-type`, `return-value`, `attr-defined` and 10 other error codes for `agentds.web.*`. The handler signatures say `Generator[tuple[str, float, str, dict], …]`, yet the final yield sends `result`, a dict holding a `PipelineState` TypedDict and a dumped Pydantic model, into a JSON component with no schema. `data_file: Any | None` relies on Gradio's `NamedString.name` compatibility shim (`gr.File` defaults to `type="filepath"`). It works today; `data_file` used as a `str` would be the forward-compatible form.

### Verdict Details
- Component Architecture: CONCERNS (dead and unassigned components; hand-copied agent list)
- State Management: CONCERNS (global shared registry; `job_id_state` never set)
- Type Safety: CONCERNS (error codes disabled wholesale; untyped JSON payload)
- Rendering Performance: CONCERNS (blocking before the first yield; concurrency 1; fake sleeps)
- Bundle Impact: PASS (N/A for Gradio; the CLI imports the web stack lazily, `cli.py:45`)

---

## 4. Backend Engineer

**Verdict**: NEEDS REWORK

*Backend Engineering Audit: Repo Health — main@d29bbd5*

**Perspective**: Backend Engineer, with the Rust/Tauri lens translated to Python (Litestar, LangGraph, LiteLLM, Redis, Docker SDK)
**Date**: 2026-11-03
**Repo Health — main@d29bbd5**: the whole codebase as it stands. There is no plan or diff, and the working tree is clean.

#### Repo Health — main@d29bbd5 Summary
Three entry points drive one `AgentDSPipeline` (LangGraph): the Litestar API, the Gradio UI and the CLI. The pipeline runs 10 synchronous agents over three components: `LLMGateway` (LiteLLM), `ArtifactStore` (local files plus a JSON registry) and `JobQueue` (Redis, falling back to an in-memory store).

**The orchestration core does not work as shipped:**
- In auto mode the pipeline re-runs the first agent until it hits LangGraph's recursion limit.
- Human-in-the-loop mode, the default everywhere, never runs the graph.
- The job IDs returned to callers never match the stored jobs.

**The API layer has its own problems:**
- It blocks its event loop.
- It accepts any API key when none are configured.
- It makes a paid LLM call on every health probe.
- It unpickles data from a Redis instance that the compose file publishes without a password.

#### Files Reviewed
| File | Backend Role | Risk |
|------|-------------|------|
| agentds/workflows/pipeline.py | LangGraph graph, job bookkeeping, human-in-the-loop (HITL) | Critical |
| agentds/web/api/webhooks.py | Litestar endpoints, module-level state | Critical |
| agentds/web/api/middleware.py | API-key auth, token-bucket rate limiting | High |
| agentds/core/job_queue.py | Job model, Redis/in-memory store (pickle) | High |
| agentds/core/llm_gateway.py | LiteLLM wrapper: retry, fallback, cost | High |
| agentds/core/artifact_store.py | Artifact files and JSON registry | High |
| agentds/agents/base.py | Agent run loop, error capture, LLM helpers | High |
| agentds/agents/data_loader.py | Ingestion from file, URL, DB and cloud | High |
| agentds/agents/api_wrapper.py | Generates the deployed `app.py` | High |
| agentds/agents/cloud_deploy.py | Docker SDK build and run | Medium |
| agentds/agents/devops.py | Writes Dockerfile, compose and k8s files | Medium |
| agentds/web/app.py | Gradio callbacks, module-level state | Medium |
| agentds/cli.py | CLI entry point | Medium |
| agentds/core/config.py | Settings, feature flags | Medium |
| agentds/core/validation.py | Path/URL/SQL validators (never called) | Medium |
| agentds/core/cache_layer.py | Redis/memory cache (pickle; not used yet) | Low |
| agentds/core/{logger,exceptions,apo}.py | Logging; error types (unused at runtime); prompt history | Low |
| agentds/integrations/{n8n,notifications}.py | Outbound webhooks and SMTP (not wired in) | Low |
| agentds/agents/{automl,feature_engineer,data_cleaning,eda_copilot,drift_monitor,optimization}.py | Temp I/O and `except` paths (skimmed) | Medium |
| docker/Dockerfile, docker-compose.yml, k8s/deployment.yaml | Runtime topology and probes | High |
| tests/test_api.py | API tests | — |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | File(s) | Failure Mode | Code Fix |
|---|-------|---------|-------------|----------|
| 1 | Auto-advance is written inside a conditional-edge router, so LangGraph throws it away | pipeline.py:440-444 `state["current_agent"] = next_agent ... return "run_agent"` | LangGraph only saves state that a node returns; a router gets a fresh copy. The next `run_agent` reads `current_agent="DataLoaderAgent"` again, so DataLoader re-runs (LLM call, parquet write and artifact each time) until `GraphRecursionError` at 25 steps. Every `human_in_loop=False` run fails this way (`agentds run --no-hitl`, API `human_in_loop:false`, Gradio with the box unticked), and the job stays QUEUED. | Keep routers pure. Add an `advance` node that returns `{"current_agent": nxt, "current_phase": ...}`, and route `run_agent → advance → run_agent/finalize`. |
| 2 | HITL mode, the default everywhere, never runs the graph, and resume starts from scratch | pipeline.py:559-567 (`if not self.config.human_in_loop: ...invoke... else: final_state = initial_state`); :608-624 (`if not state` on a StateSnapshot, then `invoke(update, config)`); no `interrupt_before` anywhere; webhooks.py:284-295; cli.py:111; app.py:304-323 | **API:** returns "started" but nothing runs. `/pipeline/action` re-enters at `preflight` with no `data_source` and ends in an error state, yet replies `{"status":"success"}`. **CLI:** prints `[OK] Pipeline completed successfully!` after zero agents. **Gradio:** fakes progress with `time.sleep(0.5)` and reports success; its approve/rerun/skip/stop handlers read `job_id_state` (app.py:67), which nothing sets, and they never call `resume`. | Always call `invoke`. Compile with `interrupt_before=["human_review"]` (or call `interrupt()` in that node). Resume with `update_state(config, {...})` then `invoke(None, config)`. Check `snapshot.next` instead of truthiness. Use a durable, shared checkpointer. |
| 3 | The job ID returned to callers never matches the stored Job, and the API and pipeline use different queues | pipeline.py:522-534 (`job_id = job_id or uuid4()`, but `create_job(...)` mints its own `Job.id`, job_queue.py:322-329); webhooks.py:110-122 keeps its own `_job_queue`, while pipeline.py:144 creates `JobQueue(self.settings)` | `GET /api/pipeline/status/{job_id}` and `POST /pipeline/cancel/{job_id}` return 404 for every pipeline. With the in-memory fallback they read a different store altogether. Cancel does nothing, because nothing checks status. `mark_running` and `update_progress` are never called, so progress is always 0 and the CLI prints a job ID that `agentds jobs` never shows. | Add `create_job(job_id=...)` or return `job.id` from `run()`. Inject one shared `JobQueue`. Update status inside nodes. Check for cancellation between agents. |
| 4 | Blocking synchronous work inside `async def` handlers freezes the API event loop | webhooks.py:174 `result = pipeline.run(...)`, :348 `result = agent.run(context)`, :142 `gateway.validate_connection()` (sync LiteLLM `completion`, polars, Docker, `httpx.get`) | One request stalls every other request for the whole pipeline or agent run (minutes). The k8s API probes (deployment.yaml:259-270) set no `timeoutSeconds`, so the 1s default applies: liveness restarts the pod in the middle of a run, and readiness drops it from the Service. | Move the work off the loop. Enqueue to RQ (a worker already exists in compose) and return 202 with the job ID. For short calls use `await anyio.to_thread.run_sync(...)` or `sync_to_thread=True`. |
| 5 | The unauthenticated `/api/health` makes a paid LLM call, with retries and fallback, on every probe | webhooks.py:141-142 `LLMGateway(settings)` then `validate_connection()`; middleware.py:238-243 exempts the path; compose:73 probes every 30s; k8s:259-270 probes every 10s and 30s on 2 replicas | The k8s probes alone make about 16 LLM completions per minute. Anyone can run up cost: the path is exempt from auth and limited only per IP (burst of 120). An LLM outage turns into probe timeouts and pod restarts. | Liveness should return a constant `{"status":"ok"}`. Readiness should ping Redis. Put the LLM check behind auth and cache its result (e.g. 5 minutes). |
| 6 | API auth accepts anything when no keys are configured, `create_api(settings)` is ignored, and request inputs reach the filesystem and network without validation | middleware.py:217-220 (`if not self.api_keys: ... return True`); :252 `settings or get_settings()` (Litestar builds the class with `app` only, webhooks.py:509-512; `settings` in `create_api` (:495) is never used); config.py:241 defaults to `[]` and compose/k8s set no `API_KEYS`; data_loader.py:244-330; pipeline.py:263, 278; validation.py is never imported | In the shipped deployments any non-empty `X-API-Key` or `?api_key=` value passes. That opens `/api/agent/run` and `/pipeline/start` to anyone, who can then: read local files (the CSV header and shape come back in `outputs.statistics`); make server-side requests via `httpx.get(url)` (SSRF); reach internal databases (`postgresql://…`) and buckets with the server's credentials; `mkdir -p` any `output_destination`; and pass `**data.config` straight into agent `extra` (e.g. DriftMonitor `reference_data` paths). The tests hide this: test_api.py:257-263 accept a 200 for an invalid key. | Reject requests when no keys are set, unless `environment=="development"` and an explicit opt-in flag is on. Register `DefineMiddleware(AuthenticationMiddleware, settings=settings)`. Validate `data_source` with `InputValidator(base_dir=upload_dir)` and a fixed `validate_url` (Rec 12). Drop `output_destination` or confine it under `output_dir`. |
| 7 | Redis is published on the host without auth, and job records are unpickled from it | docker-compose.yml:88 `"6379:6379"` (no `requirepass`); job_queue.py:226 and :252 `pickle.loads(data)` over `hvals("agentds:jobs")` | Anyone who can reach host:6379, or any pod in k8s (there is no NetworkPolicy), can `HSET agentds:jobs x <pickle>`. That gives code execution in the api and web containers on the next `GET /api/jobs` or Gradio Jobs refresh, and those containers hold the LLM keys. | Store `job.model_dump_json()` and read with `Job.model_validate_json()`. Remove the host port mapping. Enable `requirepass` or ACLs. Do the same in cache_layer.py:143 before it is wired in. |
| 8 | Agent failures are swallowed, and the job is marked COMPLETED | Every agent's `execute` catches all exceptions and calls `result.mark_failed(str(e))` (e.g. data_loader.py:133-135); base.py:446-474; pipeline.py:326-346 sets `state["error"]` only when an exception escapes; :570-574 `job.mark_completed(...)` | If DataLoader fails, the downstream agents run with no input and fail too. The pipeline still finalizes as a success, the job shows COMPLETED, and the CLI exits 0. The root cause is buried in `agent_results[...]["error"]`. | In `_run_agent_node`, add `if result.status is AgentStatus.FAILED: state["error"] = f"{agent_name}: {result.error}"` (or route to human_review in HITL mode). Make the CLI exit non-zero when `state["error"]` is set. |
| 9 | The generated `app.py` pastes raw column names into Python source | api_wrapper.py:216-218 `safe_name = name.replace(" ", "_").replace("-", "_")` and `f"    {safe_name}: float = Field(..., description=\"Feature {name}\")"`; :343 and :349-352 | Common names such as `Annual Income (k$)`, `2019_sales`, `a.b` or `class`, or any name containing `"`, produce a SyntaxError. The image builds, then uvicorn fails on import, so the deploy is broken. A crafted CSV header can inject code into the deployed service. Even names that compile break preprocessing silently: the DataFrame uses the sanitized names, `pipeline.transform` raises, and `except Exception: pass` feeds raw features to the model. | Sanitize with `re.sub(r"\W", "_", ...)` and prefix names that start with a digit or are keywords. Use `Field(alias=orig)` with `model_dump(by_alias=True)`. Emit strings with `json.dumps`. Run `compile(code, "app.py", "exec")` before saving. Never swallow transform errors. |
| 10 | The fallback chain changes the gateway's config list on every call | llm_gateway.py:178-181 returns the YAML list object itself; :229-230 `fallback.models.insert(0, model)` | Every `complete()` call, and every tenacity retry, permanently adds a model to the front of `fallback_chains["default"]` (config/llm_config.yaml:101-107). One gateway serves a whole pipeline, so the chain grows without limit. On a 429 it retries previously used and duplicate models before the configured fallbacks, and `@retry` repeats all of that 3 times. | `models = list(self.fallback_chains.get(name) or self.fallback_chains.get("default") or [default])`, then `chain = [model, *[m for m in models if m != model]]`. |

#### Recommendations
| # | Issue | File(s) | Why | Fix |
|---|-------|---------|-----|-----|
| 1 | The artifact registry is rewritten whole and non-atomically from each instance's own copy, and a corrupt file is silently reset | artifact_store.py:81, 96-103 (`open(..., "w")`, error only logged), 83-94 (returns `{}` on any error) | In compose, web and api share `../outputs` (compose:32, 67); in k8s, 2+ web replicas share a ReadWriteMany PVC. Whoever writes last drops the other processes' entries. A half-written file loads as `{}`, and the next save wipes the registry. The k8s API pods mount no volume, so their artifacts disappear on restart. | Write to a temp file and `os.replace`; add an `fcntl` lock or keep per-artifact JSON files or a Redis hash. Raise on a corrupt registry. Mount the outputs volume in the API pods. |
| 2 | The retry and fallback exception set is too narrow, and the cost calculation can fail a successful call | llm_gateway.py:183-187 and :305-317 catch only `RateLimitError, APIError, Timeout`; :273 `litellm.completion_cost(...)` | LiteLLM's ServiceUnavailable, InternalServer, APIConnection, Authentication and NotFound errors are not `litellm.APIError` subclasses. They hit `except Exception: raise`, so there is no fallback on 5xx, connection errors or a missing provider key. That matters here: compose `api` passes only the OpenAI and Anthropic keys, while the config maps DataLoader to groq and EDA to `google/...`. `completion_cost` can raise for models missing from its price table after the paid call has succeeded. `acomplete` has no fallback, and `astream` has no fallback or timeout. | Catch LiteLLM's (and OpenAI's) base exception and sort errors into retryable and fallback-able. Wrap `completion_cost` so failure gives 0.0. Retry per model instead of around the whole chain. Validate model IDs and keys at startup. |
| 3 | LLM JSON drives control flow with no schema validation | data_loader.py:154-160 (the LLM dict replaces the deterministic analysis), then :211 `analysis["source_type"]` raises KeyError; :307 runs an LLM-supplied `query` through connectorx; automl.py:156 `config["algorithms"][0]["name"]`; api_wrapper.py:469 `e['method']` | A missing or renamed key makes the agent FAIL, and Critical 8 then swallows that. The LLM can pick the loader path and run arbitrary SQL. `call_llm_structured` and `validate_sql_query` exist but are never used. | Use `call_llm_structured` with Pydantic models. Merge LLM hints over deterministic defaults. Never run LLM-written SQL without `validate_sql_query` and read-only credentials. |
| 4 | Pipeline state in the process grows without limit, and checkpoints are neither durable nor shared | webhooks.py:112, 182 and app.py:26, 290 (`_active_pipelines` is never cleared); pipeline.py:152 `MemorySaver()` | Each entry keeps a whole pipeline alive (gateway, store, agents, checkpoint history), which leaks memory. With 2 API replicas, `/pipeline/action` returns 404 when it lands on the other replica. A restart loses all HITL state, even though `checkpoint_dir` exists. | Add TTL or LRU eviction. Use a SQLite, Redis or Postgres checkpointer keyed by job_id, and rebuild the pipeline from config on resume. |
| 5 | Docker container lifecycle in CloudDeployAgent | cloud_deploy.py:170-224 | **Ports and names:** host port 8000 is fixed, so it collides with the AgentDS API and earlier deploys; the container name `agentds-{job_id[:8]}` is fixed, so a RERUN of the same job gets 409 Conflict. **Cleanup:** the container is never stopped or removed when the health check fails, and `docker.from_env()` is never closed. **Health check:** it polls `localhost:8000`, which inside the AgentDS container is not the published host port. **Socket:** compose mounts no Docker socket, so `DOCKER_LOCAL` always fails when run in a container. | Use `ports={"8000/tcp": None}` and read `container.ports`. Make names unique or remove the old container before running. Stop and remove in `try/finally`. Health-check by container IP. Call `client.close()`. |
| 6 | DevOps file generation bugs, some hidden by broken feature-flag lookup | devops.py:252 (`if "FROM" in ... and "COPY" in ...: return response.content.strip()`); :115-116 `load_text()` can return None; :190-209 nested names `.github/workflows/deploy.yml` and `k8s/{name}`, but artifact_store.py:105-110 creates only `agent_dir`; config.py:296-311 reads only the top-level `features` key | A fenced ```dockerfile reply passes the check, and `docker build` fails with "unknown instruction". A missing app-code artifact causes a TypeError. Turning on the GitHub Actions or k8s flags would raise FileNotFoundError and fail the agent. That is hidden today only because every flag under `agent_features` and `integration_features` is ignored. | Strip code fences and parse the Dockerfile. Guard against None. Call `path.parent.mkdir(parents=True, exist_ok=True)`. Support dotted flag paths. |
| 7 | Temp files are never cleaned up | data_loader.py:106-107, data_cleaning.py:138, feature_engineer.py:140-145, automl.py:146-148, drift_monitor.py:128, optimization.py:163, devops.py:109 | Every run, and every loop of Critical 1, leaves parquet and pickle copies in `temp/`, which are also copied into `outputs/` with `copy2`. Disk use grows without limit, and failed runs leave partial files. | Use a per-job `TemporaryDirectory` and clean it up in the finalize and error nodes, or write straight into the artifact store. |
| 8 | Middleware hardening | middleware.py:171 (`query_param="api_key"`), :204-208, :222 (`key in self.api_keys`), :396-400 (`dict(...headers)`); webhooks.py:498-503 (CORS `*`), :509-512 (auth runs before rate limiting) | 401 responses are not rate-limited, so keys can be brute-forced. Keys sent in query strings end up in access logs. Key comparison is not constant-time. `dict()` collapses duplicate headers such as Set-Cookie and Vary. CORS allows every origin. | Rate-limit by IP before auth. Accept keys only in headers. Use `hmac.compare_digest`. Append headers as a list. List CORS origins explicitly. |
| 9 | Redis job store design | job_queue.py:205-218 (one hash, pickled before `job.status = QUEUED`), :250-256 (`hvals` and unpickle every job, no pagination), :203 (`redis.from_url(url)` ignores `socket_timeout` and `max_connections` from config.py:109-111); compose:91 `allkeys-lru`; the RQ queue `self._queue` is never used | The stored status is `pending`. LRU eviction can drop the whole job registry at once. `GET /jobs` costs O(all jobs). An unreachable Redis can hang startup. The `worker` service sits idle. | Use one key per job plus a sorted-set index. Set `noeviction` or use a separate Redis for jobs. Pass the timeouts. Either enqueue to RQ or remove the worker. |
| 10 | Ingestion has no size limits and breaks on empty data | data_loader.py:314 (`httpx.get(url, timeout=60)` reads the whole body), :286-287 (`f.read()` reads the whole object), :347 (`null_count() / df.height`) | A large URL or object can exhaust memory. A zero-row file raises ZeroDivisionError and the agent fails. | Stream with a size cap, use polars lazy scans for cloud sources, and guard `df.height == 0`. |
| 11 | HTTP error mapping and endpoints that mislead | webhooks.py:379 (`JobStatus(status)` gives 500); :164 (`PipelinePhase(p)` gives 500 through the broad except); :199, 301 and 364 return `detail=str(e)`; :374 `limit` is unbounded; :433-440 `/config/update` reports success and does nothing; no `exception_handlers` map `AgentDSError` | Client mistakes come back as 500. Internal paths and provider messages leak to callers. Clients believe their config changed. | Register Litestar exception handlers (`ValidationError` → 400). Type the request field as `list[PipelinePhase]`. Clamp `limit`. Return 501 from config update. |
| 12 | `validate_url` and `_is_private_ip` have gaps (fix them before using them for Critical 6) | validation.py:60 (allows `file`), :275 (a literal localhost list), :289-311 (checks only 10/8, 172.16/12, 192.168/16 and 169.254/16, via `gethostbyname`) | These all pass: `http://127.0.0.2`, a hostname that resolves to 127.0.0.1, `0.0.0.0/8` addresses, IPv6 and `::ffff:` forms, and `file:///etc/passwd`. Validation is also open to DNS rebinding between check and fetch. | Check every `getaddrinfo` result with `ipaddress` (`is_private`, `is_loopback`, `is_link_local`, `is_reserved`). Drop `file`. Fetch using the IP you validated. |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| 1 | Fix EmailNotifier before wiring it in: it uses blocking `smtplib` inside `async def` with no timeout, leaves the connection open on exceptions, and doesn't escape the HTML (notifications.py:225-241). Also: `AsyncClient` is created in `__init__` (n8n.py:84, notifications.py:116), and n8n `json=` raises an uncaught TypeError on `Path` values | Integrations are not wired in yet; use `aiosmtplib` or `to_thread`, `with SMTP(..., timeout=10)` and `html.escape`. |
| 2 | `is_feature_enabled` re-reads the YAML from disk on every call (config.py:296-311) | Cache it with an mtime check. |
| 3 | `_configure_litellm` writes keys into `os.environ` for every gateway instance (llm_gateway.py:104-130) | Process-wide state where the last writer wins; pass `api_key=` per call instead. |
| 4 | `result.llm_cost = self.llm_gateway.get_total_cost()` (base.py:453) adds up all agents that share a gateway | Per-agent cost is wrong; take a snapshot before and after. |
| 5 | Gradio runs the whole pipeline in one callback, and the default `concurrency_limit=1` makes all users wait in line (app.py:257-326). `refresh_logs` reads a file nothing writes (app.py:371; `setup_logging` gets no `log_file`) | Use a background worker with polling, and pass `log_file`. |
| 6 | The API tests accept 200, 401 or 500 (test_api.py:197-216, 255-271) | This hid Critical 6; assert exact status codes and add a graph test with a fake LLM. |
| 7 | `require_admin_key` treats any key that starts with `admin_` as an admin key and raises a non-HTTP exception, which becomes a 500 (middleware.py:428-442) | It's unused; delete it or implement it properly. |
| 8 | Agent outputs carry `Path` objects (data_loader.py:124, devops.py:213-216) across JSON boundaries | Convert them to `str` at the boundary. |
| 9 | MD5 is used for checksums and cache keys (artifact_store.py:116, cache_layer.py:326) | Switch to SHA-256 to keep linters quiet. |

#### Error Path Analysis
- **LLM failure:**
  1. A LiteLLM exception reaches `LLMGateway.complete` (llm_gateway.py:305-317). On 429, APIError or Timeout it moves to the next model in the (corrupted, Critical 10) chain; anything else is re-raised. Tenacity then repeats the whole chain up to 3 times (:183-187).
  2. `BaseAgent.call_llm` doesn't handle it. The agent's `execute` catches it with a blanket `except Exception` and calls `result.mark_failed(str(e))`. The exception type, model and traceback survive only in the log.
  3. `BaseAgent.run` returns a FAILED result and raises nothing. `_run_agent_node` does not set `state["error"]`, so the run reaches `finalize` and `job.mark_completed`.
  4. What the user sees: CLI prints `[OK]` and exits 0; Gradio says "completed successfully"; the API can't show it at all, because status returns 404 (Critical 3).
- **Graph-level exceptions** (the `GraphRecursionError` from Critical 1, checkpoint serialization errors): `pipeline.run` has no try, so the job stays QUEUED forever. The API answers 500 with raw `str(e)`, and the CLI exits 1.
- **LLM JSON parse failures** are swallowed and replaced with a default plan, and nothing in the result says so: data_loader.py:161 (`except Exception: pass`), data_cleaning.py:296, eda_copilot.py:280, automl.py:236, feature_engineer.py:223, api_wrapper.py:187.
- **Defaults that mislead:** drift_monitor.py:227-229 returns drift `0.0` when the calculation fails, which looks the same as "no drift", so alerts are suppressed. data_cleaning.py:381 and eda_copilot.py:219 skip failed steps silently. automl.py:374 `except: pass` drops roc_auc.
- **Persistence:**
  - `ArtifactStore._save_metadata` only logs on failure, so the caller gets an Artifact the registry never saved.
  - `_load_metadata` turns any error into `{}`, and the next save erases the registry.
  - apo.py:124-126 resets prompt history on a load error, and the next `_save` overwrites the file.
- **Redis:** CacheLayer swallows errors, which is fine for a cache. RedisQueue catches nothing, so connection errors become bare Litestar 500s. `JobQueue._init_backend` quietly falls back to memory if Redis is down at startup, and that pod then never sees jobs stored in Redis (split brain).
- **API boundary:**
  - Handlers catch `Exception` and return 500 with `str(e)`, so bad enum values come back as 500.
  - The `AgentDSError` hierarchy, with its codes and `to_dict()`, is never raised on these paths, and no exception handler maps it to HTTP.
  - `InvalidAPIKeyError` would only be raised by guards, and the guards aren't used.

#### Resource Lifecycle Audit
| Resource | Allocated In | Cleaned Up In | Error Path Cleanup | Risk |
|----------|-------------|---------------|-------------------|------|
| Pipeline objects (gateway, store, agents) | webhooks.py:171, 182; app.py:286, 290 | Never | None | High (memory leak) |
| LangGraph MemorySaver checkpoints | pipeline.py:152 | When the process ends | Lost on restart | Medium |
| Temp parquet, pickle and devops dirs | data_loader.py:106, feature_engineer.py:140-145, automl.py:146, devops.py:109, others | Never | Partial files are left behind | High (disk) |
| Docker image and container | cloud_deploy.py:176-196 | Never | Left running when health fails; dangling layers on build failure | High |
| Docker client | cloud_deploy.py:166 | Never closed | — | Low |
| Artifact registry file | artifact_store.py:100 | Rewritten in place | A half-written file resets the registry | High |
| Redis connection pools | job_queue.py:203, cache_layer.py:131 (a new one per `JobQueue`, e.g. pipeline.py:144) | Never closed | — | Low–Medium |
| LLMGateway on each health request | webhooks.py:141 | Garbage collection | Changes `os.environ` | Low |
| httpx AsyncClient | n8n.py:84, notifications.py:116 | `close()` / `__aexit__` | Leaks if the caller skips close (not wired yet) | Low |
| SMTP connection | notifications.py:227-234 | `quit()` on success only | Leaks on exception; no timeout | Medium (not wired yet) |
| File handles | Every `open()` | `with` blocks | OK | Low |
| Rate-limit buckets and MemoryCache | middleware.py:99; cache_layer.py:63 | Cleared every 5 min; capped at 1000 entries | OK | Low |

#### Verdict Details
- Error Handling: CONCERNS. Failures are swallowed and reported as success (Critical 8); exception types are lost at the agent boundary; the exception hierarchy is unused; the API returns raw `str(e)` with 500s.
- Concurrency Safety: CONCERNS. Blocking calls inside async handlers (Critical 4); module-level pipeline dictionaries; a shared mutable fallback list (Critical 10); a registry where the last writer wins across processes (Rec 1).
- Resource Management: CONCERNS. Temp files, Docker containers and in-process pipelines are never released.
- IPC Correctness (HTTP, Redis, Gradio): CONCERNS. Job IDs don't match (Critical 3); no-op actions return success (Critical 2); pickle crosses the Redis boundary (Critical 7); the generated code boundary is unsafe (Critical 9).
- File System Safety: CONCERNS. Unvalidated `data_source` and `output_destination` (Critical 6); `validation.py` is never called. Path traversal through `job_id` can't be reached: job IDs are server-generated UUIDs and `cleanup_job` is never called.
- Checklist items with no analogue here:
  - Mutex held across await, and lock-ordering deadlocks: there are no locks or asyncio primitives; the nearest equivalent is blocking inside async, covered in Critical 4.
  - serde casing and enum tag format: request/response models are snake_case Pydantic on both sides, enums are `str` enums, and there is no TypeScript client.
  - Sidecar crash and restart: the only spawned processes are Docker containers, covered in Rec 5.
  - Clippy `#[allow]` ban: no equivalent; `ruff check .` passes with the repo's config.

#### Confidence notes
- **Critical 1:** this relies on LangGraph semantics (a router gets a fresh read of state, and only node returns are saved). LangGraph isn't installed here, so I did not run it. I'm confident in the mechanism, but it's unexecuted.
- **Critical 4:** the pod restarts depend on k8s's default `timeoutSeconds=1` and on LLM latency. That the event loop blocks during a pipeline run is certain from the code.
- **Critical 7:** real-world exposure depends on host firewalling. The code-level path (pickle loaded from unauthenticated Redis) is certain.
- **Critical 9:** assumes column names reach `fe_plan` (written by the LLM) or `feature_names` without being normalized. The LLM might rename them, but nothing in the code guarantees it.
- **Rec 2:** the LiteLLM exception hierarchy (APIError as a sibling of the 5xx/connection/auth exceptions rather than their parent) and `completion_cost` raising for models missing from its price table come from my knowledge of LiteLLM, and may differ by version. I believe model IDs like `groq/llama-4-scout` and `google/gemini-2.5-flash` are not valid LiteLLM routes, but I did not check them against the current provider registry.
- **Not flagged:** Litestar handles a `Response` returned from a handler annotated with a model type, and Gradio 5's `gr.File` value still has `.name` (NamedString).
- **Earlier review:** REVIEW_FEEDBACK.md scores "Robust Input Validation" 8/10, but `validation.py` is never imported outside `core/__init__.py`, so that claim doesn't hold. Its pickle finding is confirmed for `job_queue.py`; `cache_layer.py` is not used yet.

---

## 5. Structural Engineer

**Verdict**: NEEDS REWORK

*Structural Audit: Repo Health — main@d29bbd5*

**Perspective**: Systems/Structural Architect
**Date**: 2026-11-03
**Plan**: Repo Health — main@d29bbd5. There is no plan; I audited the codebase as it stands in /home/user/AgentDS.

#### Repo Health Summary
The static import graph is clean. An AST scan of every `agentds` import finds no cycles, and the layers point the right way (core ← agents ← workflows ← web/cli, with integrations depending only on core). `ruff check .` passes. The structural problems are in the *runtime* contracts, which have no types behind them:
- the job/HITL contract between the web layer and the workflow
- the routing state inside the LangGraph pipeline
- the string-keyed agent-to-agent handoffs
- shared mutable state: the LLM gateway, the artifact index and three separate job queues

On top of that, about 1.9k LOC (around 16% of the package) has no runtime importer: `apo`, `cache_layer`, `validation` and `integrations/`.

#### Files Reviewed
| File | Structural Role | Layer | Risk |
|------|----------------|-------|------|
| agentds/workflows/pipeline.py | Orchestrator (LangGraph graph, HITL, job record) | workflow | High |
| agentds/web/api/webhooks.py | Whole REST API: models, global state, handlers, app factory | web | High |
| agentds/web/app.py | Gradio UI and its handlers (one 390-line closure) | web | High |
| agentds/web/api/middleware.py | Auth and rate-limit middleware | web | Med |
| agentds/agents/base.py | BaseAgent, AgentContext, AgentResult (extension point) | agents | Med |
| agentds/agents/{data_loader,data_cleaning,eda_copilot,feature_engineer,automl}.py | Build agents (handoff via string keys) | agents | High |
| agentds/agents/{api_wrapper,devops,cloud_deploy,drift_monitor,optimization}.py | Deploy and Learn agents (large embedded templates) | agents | Med |
| agentds/core/llm_gateway.py | LiteLLM gateway shared by all agents | core | High |
| agentds/core/job_queue.py | Job registry (in-memory or Redis hash) | core | High |
| agentds/core/artifact_store.py | File store with a single JSON index | core | Med |
| agentds/core/config.py | Settings, YAML loaders, feature flags | core | Med |
| agentds/core/{apo,cache_layer,validation,exceptions}.py | Mostly orphaned subsystems | core | Med |
| agentds/integrations/{n8n,notifications}.py | Event clients, nothing imports them | integrations | Low |
| agentds/__init__.py, core/, agents/, web/, workflows/ `__init__` | Public API surface, eager imports | pkg | Med |
| agentds/cli.py | Composition root #1 | entry | Med |
| docker/{Dockerfile,docker-compose.yml,k8s/deployment.yaml}, config/*.yaml | Deploy topology and config sources | deploy | Med |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | Files Involved | Structural Impact | Fix |
|---|-------|---------------|-------------------|-----|
| C1 | The pipeline's job ID never reaches the Job record, and each entry point builds its own JobQueue | pipeline.py:522-534 (`job_id = job_id or str(uuid.uuid4())`, then `self.job_queue.create_job(name=…, task=…, kwargs=…)` with no id; `Job.id` gets its own uuid4 at job_queue.py:48). Separate `JobQueue(...)` at pipeline.py:144, webhooks.py:121 and app.py:40. job_queue.py:200-201 does `from rq import Queue`, but `rq` is missing from pyproject (it is only in requirements.txt:68), so job_queue.py:293-298 silently falls back to in-memory | `/api/pipeline/start` returns a job ID that `/api/pipeline/status/{id}` (webhooks.py:211) and `/api/jobs` can never find, so the client gets a 404. With the in-memory backend the Gradio Jobs tab and `agentds jobs` never see pipeline jobs | Add `id` to `create_job` and pass the job ID through. Inject one JobQueue (app state) into the pipeline. Declare `rq` in pyproject |
| C2 | The human-in-the-loop contract between web and workflow is not implemented | pipeline.py:559-567 (`if not self.config.human_in_loop: …invoke… else: final_state = initial_state`). HITL is the default in the API (webhooks.py:42), Gradio (app.py:111-114) and the CLI (cli.py:98), so the graph never runs. No `interrupt_before` on compile (pipeline.py:557/585/604), so human_review → handle_action falls through with `user_action=None` and routes back to the same agent (370-397, 448-458). `resume()` (604-624) invokes a thread that never ran, so it restarts at preflight with no data_source. Gradio buttons (app.py:328-353) never call resume, and `job_id_state` (app.py:67) is never written. The pipeline registries are per-process dicts (webhooks.py:112, app.py:26), yet the k8s API runs `replicas: 2` (deployment.yaml:222) | In its default mode every entry point does nothing. The CLI still prints "[OK] Pipeline completed successfully!" (cli.py:111), and `/pipeline/action` returns success while the resumed run errors | Compile once with a checkpointer and `interrupt_before=["human_review"]`, and always invoke. Resume on the same thread with a state update. Move checkpoints and the pipeline registry to Redis or SQL instead of module dicts. Route all three UIs through one pipeline service |
| C3 | Moving to the next agent happens inside a conditional-edge router | pipeline.py:440-444 (`state["current_agent"] = next_agent` inside `_route_after_agent`). `_run_agent_node` (295-346) never advances | LangGraph discards state mutations made in routers. Non-HITL runs (`--no-hitl`, API `human_in_loop=false`) re-run DataLoaderAgent until GraphRecursionError (default limit 25) | Advance in a node (return `current_agent`/`current_phase` from `_run_agent_node` or a dedicated `advance` node). Keep routers pure |
| C4 | Agent-to-agent data handoff is an untyped string lookup, and FeatureEngineer and EDA disagree on keys | feature_engineer.py:109-119 prefers `previous_results.get("EDACopilotAgent")`, then requires `outputs.get("data_path")`. EDACopilotAgent's outputs (eda_copilot.py:142-148) have no `data_path`. Selection is by presence, not success, and failed results are stored too (pipeline.py:330-331) | Every BUILD run that includes EDA fails at FeatureEngineerAgent ("No data_path in previous output"), so AutoML and the Deploy phase never run | Add typed per-agent output models, or carry a `DatasetRef` in pipeline state. Resolve inputs from the declared `input_types` of results that completed |
| C5 | A live polars DataFrame goes into `AgentResult.outputs`, which crosses serialization boundaries | data_cleaning.py:168 (`"dataframe": df_cleaned`) → pipeline.py:331 `result.model_dump()` into graph state checkpointed by MemorySaver (pipeline.py:54,152) → `final_outputs` → `Job.result` (pipeline.py:574) → JSON at webhooks.py:230 and `gr.JSON` at app.py:323 | Checkpoint writes and HTTP/Gradio serialization fail. This is hidden today behind C2–C4 and will surface as soon as those are fixed | Allow only JSON-safe references (paths, artifact IDs) in outputs, and enforce that with a validator on `AgentResult.outputs` |
| C6 | The LLM fallback chain is a shared config list that every call mutates | llm_gateway.py:178-181 returns `FallbackChain(models=models)`, where `models` is the list object inside `self.fallback_chains` (loaded from llm_config.yaml:101-107). llm_gateway.py:230 `fallback.models.insert(0, model)` mutates it in place | One gateway is shared by all agents (pipeline.py:142,160-164), so the "default" chain grows by one entry per call. On a rate-limit or API error it then tries other agents' primary models and repeats the failing one. Growth is unbounded in long-lived processes | Copy the list: `FallbackChain(models=[model, *chain])`. Freeze the loaded config (tuples) |
| C7 | The validation module is orphaned, so HTTP input reaches agents unvalidated | core/validation.py (488 LOC: `sanitize_path`, `validate_url`, …) has no runtime importer; only core/__init__.py:43-49 imports it. webhooks.py:174-179 and 335-344 put `data_source`, `output_destination` and `**data.config` straight into `AgentContext.extra`. data_loader.py:310-314 does `httpx.get(url)`, 240-245 reads any local path, and 307 runs the LLM-supplied `analysis.get("query")`. pipeline.py:277-278 does `Path(output_dest).mkdir(parents=True)`. With no keys configured, middleware.py:217-220 accepts any non-empty key | SSRF, arbitrary file read and arbitrary directory creation through the REST API | Validate at the web boundary (request models call the `validation.*` functions), restrict data roots, and reject unknown `config` keys. The security lens should rate the exposure |

#### Recommendations
| # | Issue | Why | Fix |
|---|-------|-----|-----|
| R1 | There is no execution boundary. Async handlers run pipelines and agents synchronously (webhooks.py:157-179, 308-348). JobQueue is a registry, not a queue: `RedisQueue.enqueue` writes a pickled hash and never uses `self._queue` (job_queue.py:204-218). The compose `worker` (docker-compose.yml:159-179) has nothing to consume | A long run blocks the event loop, including `/api/health`, so k8s liveness kills pods mid-run. `/pipeline/cancel` changes a record that nothing reads | Put a `PipelineService` behind the web layer that enqueues to RQ (or similar) and is driven by the worker. Handlers only enqueue and read status |
| R2 | The health endpoint is coupled to a paid external LLM. webhooks.py:137-147 builds a new LLMGateway and calls `validate_connection()`, which makes a real completion (llm_gateway.py:457-478) | k8s API probes run every 10s and 30s on at least 2 replicas (deployment.yaml:259-270; HPA min 2) with the default 1s timeout. That is about 16 billed calls a minute, it blocks the loop, and probes can flap | Liveness checks the process only. Readiness checks config only. Move the LLM check to an opt-in `/api/health/deep` |
| R3 | The composition root ignores the settings it is given. `create_api(settings)` assigns `settings` and never uses it (webhooks.py:495). The middleware re-reads `get_settings()` (middleware.py:252,322), and handlers use a module-global `get_dependencies()` (webhooks.py:110-122) | Embedders and tests (test_api.py:181/243/299, e.g. `rate_limit_per_minute=5`) silently get environment config. Auth and rate-limit config can't be injected | Store settings and the job queue in `app.state`. Pass settings to middleware via `DefineMiddleware(..., settings=settings)` |
| R4 | Eager `__init__` imports invert the layering at load time. agentds/__init__.py:30-36 imports `workflows.pipeline`, so `import agentds.core.exceptions` loads every agent, sklearn and langgraph. core/__init__.py:16-49 loads litellm. web/__init__.py:9 imports the Gradio app, so the API process (`python -m agentds.web.api.webhooks`, docker-compose.yml:56, deployment.yaml:234) imports gradio and loads webhooks twice (runpy) | Startup cost, low layers can't be imported on their own, and module-level state is duplicated | Use lazy PEP 562 `__getattr__` exports. Keep the root and core `__init__` minimal. Don't import app.py from web/__init__ |
| R5 | Orphaned or duplicated subsystems: core/apo.py (525 LOC) duplicates OptimizationAgent's own critique/rewrite and agentlightning wrapper (optimization.py:333-391, 486-546 vs apo.py:255-500); cache_layer.py (332) and integrations/ (579) have no importer; most of exceptions.py (510) is unused because agents raise ValueError. OptimizationAgent optimizes placeholder prompts (optimization.py:299-314) and never writes them back | Code that is drifting, docs that claim the wiring exists (docs/ARCHITECTURE.md), and a "self-improvement" loop with no write-back path | Wire each piece in or delete it. Keep one APO implementation in core, used by OptimizationAgent with APOSettings |
| R6 | Agents duplicate boilerplate the base class already provides. Six agents copy the "regex `\{.*\}` + json.loads + default" block (api_wrapper.py:180-186, automl.py:229-235, data_cleaning.py:288-295, data_loader.py:154-160, eda_copilot.py:272-279, feature_engineer.py:215-222), while `BaseAgent.call_llm_structured` (base.py:223-288) has no callers. Sync and async variants are copied (base.py:177-365). There are 10 private `_format_approval_message` methods while `get_approval_message` (base.py:494) goes unused | A change to parsing or error policy has to be made in about 10 files | Add one `call_llm_json(prompt, model, default)` helper with typed response models. Override `get_approval_message` |
| R7 | Agent identity and phase order are hard-coded in 6+ places: pipeline.py:109-126, agents/__init__.py:64-75, app.py:179-181, cli.py:128-138, optimization.py:234-241 and 302-312, llm_config.yaml `agent_llm_mapping`. pipeline_config.yaml:70-97 is never loaded (`get_pipeline_config` has no caller). The `name`/`phase` attributes on each agent (e.g. devops.py:39-41) are unused | Renaming or adding an agent touches every one of these places | Build the registry and phase order from class attributes, or load pipeline_config.yaml once, so there is one source of truth |
| R8 | Dependencies are injected two ways. `AgentContext.llm_gateway` and `.artifact_store` (base.py:106-107) are never read by agents; every agent uses `self.*` instead. cli.py:158-164 and webhooks.py:331-339 each build two gateways and two stores per run and ignore one of each | A misleading extension contract and duplicate construction | Drop these fields from AgentContext, or make agents use the context only |
| R9 | Shared mutable state in three places. `llm_cost = self.llm_gateway.get_total_cost()` (base.py:453) reads the cumulative cost of a shared gateway that is never reset. `LLMGateway.__init__` writes process-wide `os.environ` (llm_gateway.py:103-130). Each ArtifactStore loads `.artifacts.json` once and rewrites the whole file on every save (artifact_store.py:80-104, 203-204) | Each agent's reported cost includes all earlier agents. Gateways overwrite each other's env. Index entries are lost when stores in different processes or replicas write (web and api share ../outputs in compose; 2 or more web pods share an RWX PVC) | Report per-agent usage deltas. Pass keys per call. Use per-artifact metadata files or a Redis/DB index |
| R10 | Feature flags and config are looked up by string and silently resolve to false. `is_feature_enabled("github_actions")` and `("kubernetes_configs")` (devops.py:102-105) search only the top-level `features:` block (config.py:296-311), but those keys live under `agent_features.devops` (feature_flags.yaml:129-138), so they are always False. YAML paths are relative to the working directory (config.py:257-259). `human_in_loop` has 4 owners (config.py:247, pipeline.py:52, pipeline_config.yaml:39, feature_flags.yaml:14), and the k8s `HUMAN_IN_LOOP` setting has no effect on the API | Documented features are silently off, and running from another directory silently drops all config | Add a typed flag accessor that does nested lookup and raises on unknown names. Resolve the config dir from the package or an env var. Give each setting one owner |
| R11 | The web layer reaches into workflow internals and fakes behavior. app.py:305-319 reads `pipeline.PHASE_AGENTS` and the private `pipeline._agents`, then simulates progress with `time.sleep(0.5)`. The `llm_preset` input is ignored (app.py:263). The Logs tab reads `log_dir/agentds.log` (app.py:371), but `launch_app` configures no log file (app.py:438) | The UI doesn't reflect real execution. The root cause is that the pipeline has no event interface, even though n8n.py:23-33 already defines those events | Add a `PipelineObserver` protocol in workflows. Have JobQueue progress, Gradio, n8n and notifications subscribe to it |
| R12 | The orchestration layer has no tests: nothing in tests/ imports AgentDSPipeline. The API tests are `@pytest.mark.integration`, which pyproject `addopts` excludes, so CI never exercises the web→workflow boundary | This gap is how C1–C4 shipped | Add graph-level tests with stub agents (no LLM) covering HITL interrupt and resume, the non-HITL traversal, and the job-ID round trip |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| N1 | Split webhooks.py into `schemas.py`, `routes/` and `deps.py`, and rename it (it is the whole REST API, not webhooks) | Cohesion and discoverability |
| N2 | Move generated-code templates out of Python strings (api_wrapper.py:207-440, devops.py:231-695) into package data (Jinja2 is already in requirements.txt). Share the model path constants (api_wrapper.py:288-289 vs devops.py:121-133). Remove the duplicate requirements generators (api_wrapper.py:440 vs devops.py:468) | Hidden coupling: DevOps knows the API template's file layout |
| N3 | Public API gaps: `AgentContext` (the parameter of `BaseAgent.execute`) is not exported from `agentds` or `agentds.agents`. `PipelinePhase` (needed for `PipelineConfig`) is not exported from `agentds.workflows`. `AGENT_REGISTRY` and `get_agent` are missing from `__all__`. `require_api_key` is exported but unused | Callers import from submodules, which locks in internals |
| N4 | Remove dead fields: `PipelineConfig.auto_approve_low_risk` and `max_retries` (`retry_count` is incremented at pipeline.py:376 but never checked), `approval_message` (every agent builds one, nothing reads it), and BaseAgent `input_types`/`output_types`/`complexity` | Contract noise; reruns are unbounded |
| N5 | Stop the Settings validator from creating directories on construction (config.py:269-279) | Side effects in the working directory on any `get_settings()`, even `agentds --help` |
| N6 | Reconcile pyproject and requirements.txt (`rq`, `docker`, `lightgbm` and others are only in the latter, and Docker builds from the latter). Drop declared but unused hard dependencies (mlflow, duckdb, llama-index, pydantic-ai: none are imported) | Two dependency manifests disagree, and installs are heavier than needed |

#### Dependency Map
```
                               agentds/cli.py   (lazy imports)
                         /            |              \
             web/app.py (Gradio)   web/api/webhooks.py ──► web/api/middleware.py
               │   [!] reads           │      └──(bypasses workflow)──► agents.AGENT_REGISTRY, agents.base.AgentContext
               │   pipeline._agents    │
               ▼                       ▼
                  workflows/pipeline.py
                          │
                          ▼
     agents/__init__ (AGENT_REGISTRY) ──► 10 agents ──► agents/base.py
                          ┆ implicit string edges via previous_results["<Agent>"].outputs["<key>"]:
                          ┆ Cleaning→Loader; EDA→Cleaning|Loader; FE→EDA|Cleaning|Loader [!] EDA has no data_path;
                          ┆ AutoML→FE; APIWrapper→AutoML,FE; DevOps→APIWrapper,AutoML,FE; CloudDeploy→DevOps;
                          ┆ Drift→Cleaning; Optimization→Drift
                          ▼
     core:  config ◄── logger ◄── {llm_gateway, artifact_store, job_queue}      (acyclic, never imports upward)
            no runtime importer: apo, cache_layer, validation, most of exceptions
     integrations/{n8n, notifications} ──► core.logger                          (no inbound edges)

 Load-time inversions (concerning):
   agentds/__init__.py ──► workflows.pipeline, agents.base   (any `import agentds.core.x` loads the top layer)
   agentds/web/__init__.py ──► web.app (gradio)              (the API process always imports Gradio)
 Runtime state edges (not imports, concerning):
   pipeline ─JobQueue()   webhooks ─JobQueue()   app ─JobQueue()     → 3 independent queues (in-memory in pip installs)
   webhooks._active_pipelines  /  app._active_pipelines              → per-process HITL registries
   all agents ─► one shared LLMGateway (cost counter, mutable fallback lists, os.environ)
   every ArtifactStore ─► whole-file rewrite of outputs/.artifacts.json
```

#### Boundary Crossing Analysis
| Crossing | From -> To | Clean? | Notes |
|----------|-----------|--------|-------|
| CLI → workflow | cli.py:87-109 → `AgentDSPipeline.run` | No | Defaults to HITL, so nothing runs while it reports success (C2) |
| CLI → single agent | cli.py:143-170 → AGENT_REGISTRY | Partial | Same composition logic as webhooks; builds two gateways and two stores (R8) |
| REST → workflow | webhooks.py:171-185, 285-289 | No | Sync work in an async handler; job ID mismatch; in-process registry (C1, C2, R1) |
| REST → agent (bypasses workflow) | webhooks.py:313-348 | No | Unvalidated `**config` goes into `extra` (C7). Every agent except DataLoader needs `previous_results`, so the rest fail when run alone |
| Gradio → workflow | app.py:286-323 | No | Reads private `_agents`, fakes progress, HITL not wired (R11, C2) |
| workflow → agents | pipeline.py:154-164, 308-331 | Partial | Dynamic registry is fine; state goes through a `model_dump` dict round-trip that lets a DataFrame leak (C5) |
| agent → agent | `previous_results` string keys | No | Implicit contract that is already broken (C4) |
| agents → core | base.py:21-24 | Yes | Clean downward dependency |
| workflow → job_queue | pipeline.py:144, 526-575 | No | Private queue instance; job ID not passed through (C1) |
| core → YAML config | config.py:257-311 | Partial | Paths relative to working dir; flat flag lookup misses nested keys (R10) |
| core → process env | llm_gateway.py:103-130 | No | Global side effect from a constructor (R9) |
| health → LLM provider | webhooks.py:137-147 | No | Liveness depends on a paid external call (R2) |
| deploy → code | docker-compose.yml:159-179 (worker), deployment.yaml:222 (API replicas: 2) | No | The worker has nothing to consume; replicas break in-process HITL state |

#### Verdict Details
- Dependency Direction: CONCERNS. The static DAG is correct (core never imports agents, workflows or web, and there are no cycles), but the root and `web/__init__` eager imports invert it at load time.
- Module Coupling: CONCERNS. One gateway with mutable state is shared by everything, the artifact index is shared, there are three job queues, and agents couple to each other through string keys.
- Cohesion: CONCERNS. Two APO implementations; webhooks.py is the whole API; templates live inside agent code; about 1.9k LOC has no importer.
- Layer Boundaries: CONCERNS. The web layer bypasses the workflow, reads private state and passes unvalidated input inward; the validation layer is orphaned.
- Change Propagation: CONCERNS. Agent names and order live in 6+ places, JSON parsing is copied 6 times, and `human_in_loop` has 4 owners.

**Checklist items with no analogue here**
- "Missing adapter for backend switch" (Claude vs OpenCode): no analogue. The single LiteLLM gateway is the provider abstraction and is fine apart from C6.
- "Monorepo boundaries / apps/common": no analogue; this is a single package.
- "Back-door circular dependency through types/constants": none found in imports. The real coupling is the string-key data contract (C4).

**Confidence notes**
- **C3** depends on LangGraph discarding state mutations made in conditional-edge routers. That is how `StateGraph` branch readers behave in 0.2+/1.x as I understand them; I did not run it, because dependencies are not installed.
- **C2 (resume path)** assumes `get_state` on a thread that never ran returns an empty but truthy snapshot, and that `invoke` then starts a fresh run at preflight. This matches LangGraph semantics but was not executed.
- **C5 checkpoint crash** depends on the langgraph-checkpoint version. Version 2.x+ serializes with msgpack and no pickle fallback, which raises TypeError; older JSON serializers may degrade silently instead. The HTTP and `gr.JSON` serialization failure does not depend on the version. Today it is hidden behind C2–C4.
- **C7**: it is certain that `validation.py` has no runtime importer. How much data actually leaks back through outputs is for the security lens to judge.
- **R2**: whether probes actually time out depends on provider latency. The per-probe billed call is certain whenever API keys are configured.
- **R9 (artifact index lost updates)**: one could argue this is Critical under the multi-replica deploy. I rated it a Recommendation because the artifact files survive and only index entries are lost.
- LOC figures come from `wc -l` and are approximate.

---

## 6. Repo Maintainer

**Verdict**: NEEDS REWORK

*Repo Maintenance Audit: Repo Health — main@d29bbd5*

**Perspective**: Repo Maintainer
**Date**: 2026-11-03
**Scope**: Repo Health — main@d29bbd5 (the whole tree at /home/user/AgentDS; there is no plan or diff)

#### Repo Health — main@d29bbd5 Summary
`ruff check` passes and CI is green. That green status rests on suppressions, a gutted mypy config and a security job that always succeeds, and it hides a guaranteed NameError in the deploy phase. The README, CONTRIBUTING and the deploy docs describe setup steps, commands and features that do not match the code. About 1,900 lines of "shipped" modules (validation, cache, APO, integrations) have no production callers, and they carry most of the test suite.

#### Files Reviewed
| File | Convention Status | Risk |
|------|------------------|------|
| pyproject.toml | Lint config hides bugs (F821 ignore, 13 mypy codes off); deps drift | High |
| requirements.txt | Out of sync with pyproject; mixes runtime, dev and optional deps; Docker builds from it | Med |
| .github/workflows/ci.yml, release.yml | Security job cannot fail; no format, coverage or version gates | High |
| .github/RULESET_GUIDE.md, CODEOWNERS, dependabot.yml | Check names and owner handle don't match reality | Med |
| CONTRIBUTING.md, SECURITY.md, README.md | Broken setup steps, a nonexistent `develop` branch, placeholder security contact | High |
| docs/API_REFERENCE.md, USER_MANUAL.md, DEPLOYMENT.md, CHANGELOG.md | Several examples and endpoints don't match the code | Med |
| .env.example + agentds/core/config.py | The documented copy step likely crashes Settings | High |
| agentds/agents/api_wrapper.py | f-string template always raises NameError | High |
| agentds/workflows/pipeline.py, web/api/webhooks.py, cli.py, web/app.py | HITL path is a no-op; job IDs mismatch; dead UI | High |
| agentds/agents/base.py + 6 agents | Duplicated JSON parsing; unused helpers | Med |
| agentds/core/{validation,cache_layer,apo}.py, integrations/ | No production callers | Med |
| tests/* | 182 tests; ~62% target unwired modules; API tests accept 500 | High |
| docker/Dockerfile, docker-compose.yml, k8s/ | Applying the k8s dir overwrites the secret; image never published | High |
| REVIEW_FEEDBACK.md, REVIEW_SUMMARY.md | Stale bot review at the root with false claims | Low |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | File(s) | Convention Violated | Fix |
|---|-------|---------|-------------------|-----|
| 1 | **APIWrapperAgent can never generate code.** Inside the `code = f'''...'''` template (starts at :222), `{field: [getattr(data, field)] for field in data.model_fields.keys()}` is parsed as a replacement field, so it evaluates `field` and raises `NameError` on every call. I reproduced this f-string behaviour in isolation. The error is caught at :158-160, so the agent always ends FAILED and the Deploy phase cannot succeed. The bug is hidden three ways: ruff `"agentds/agents/api_wrapper.py" = ["F821"]  # False positive on list comprehension variable`, mypy `name-defined` disabled, and the `# type: ignore[name-defined]` on those lines, which is just text inside the generated file. No test calls `_generate_api_code`. | api_wrapper.py:343, :382; pyproject.toml:143, :162 | No lint suppressions; suppressions must not mask real errors | Escape to `{{field: ...}}`. Delete the F821 ignore and re-enable `name-defined`. Add a test that renders the template and `compile()`s it. |
| 2 | **The README Quick Start step likely crashes every entry point.** README.md:98, CONTRIBUTING.md:73 and USER_MANUAL.md:51 all say `cp .env.example .env`. .env.example:134 has `API_KEYS=key1,key2,key3`. `Settings.api_keys: list[str]` (config.py:241) is read from `.env` (config.py:212). pydantic-settings JSON-decodes list fields before the `mode="before"` validator (config.py:263-268) runs, so `Settings()` raises `SettingsError`. Separately, LLM keys in `.env` never reach `LLMSettings`: it has no `env_file` (config.py:24-28) and is built through `default_factory` (config.py:251). test_core.py:44 passes the string as a kwarg, so it never exercises the env path. | .env.example:134; config.py:24-28, 212, 241, 251, 263-268 | Docs/examples must match code | Use `Annotated[list[str], NoDecode]` (or a str field plus a parser). Make nested settings read `.env` (env_file, or `env_nested_delimiter`). Add tests that load from env and a dotenv file. |
| 3 | **The default human-in-the-loop path does nothing and reports success.** When `human_in_loop` is true, `run()` sets `final_state = initial_state` and never invokes the graph. True is the default in PipelineConfig (pipeline.py:52), the API (webhooks.py:42), the UI checkbox (app.py:111) and the CLI unless `--no-hitl` is passed. No LangGraph `interrupt` exists anywhere, so `resume()` (:608-622) invokes a thread that never ran. The CLI then prints `[OK] Pipeline completed successfully!` (cli.py:111) and the UI fakes progress (`# Simulate agent execution` / `time.sleep(0.5)`, app.py:317-322). The README's headline command (README.md:110-113) takes this path. | pipeline.py:560-567, 608-622; cli.py:111; app.py:317-322 | Docs must match code; no stub left looking like a feature | Implement `interrupt_before=["human_review"]` and actually invoke the graph. Until then, fail loudly or default HITL to off. Add pipeline tests with a mocked gateway. |
| 4 | **API pipeline status is never found.** `/pipeline/start` makes a fresh uuid (webhooks.py:172) and passes it as `job_id`. `run()` records the job in its own `JobQueue` instance (pipeline.py:144) under an auto-generated id (pipeline.py:526; job_queue.py:48). So `/pipeline/status/{job_id}` (webhooks.py:211), `/jobs/{id}` and cancel always return 404 or 400. The n8n workflow polls exactly this endpoint (N8N_GUIDE.md:138). test_api.py:203 accepts `[400, 404, 500]`, and no test starts a job and then checks its status. | webhooks.py:121, 172, 211; pipeline.py:144, 526 | Tests must assert behaviour, not tolerate errors | Pass `job_id` into `create_job` / `Job(id=...)` and inject one shared JobQueue. Add a start-then-status test. |
| 5 | **The documented k8s deploy is broken.** DEPLOYMENT.md:111-117 creates the real `agentds-secrets`, then runs `kubectl apply -f docker/k8s/`. That re-applies the Secret in deployment.yaml:35-43 (`OPENAI_API_KEY: "replace-with-actual-key"`) and overwrites the real keys. Both Deployments pull `ghcr.io/mlvpatel/agentds:latest` (deployment.yaml:145, 233), and no workflow builds or pushes that image. README.md:353 points to the same command. | docker/k8s/deployment.yaml:35-43, 145, 233; DEPLOYMENT.md:111-117 | Docs must match infra | Move the Secret to a `secret.example.yaml` outside the applied directory. Add an image-publish job, or document build, push and image override. |

#### Recommendations
| # | Issue | Why | Fix |
|---|-------|-----|-----|
| 1 | **CI does not enforce what it claims.** `bandit ... -o bandit-report.json \|\| true` and `safety check --json \|\| true` (ci.yml:72, 75); the report is never uploaded. mypy disables 13 error codes for every package, plus `strict_optional=false` (pyproject.toml:146-166). Yet README.md:10 shows a "Type checked" badge and CONTRIBUTING.md:138 says "All functions must have type hints". There is no `ruff format --check`, and 26 of 42 .py files would be reformatted. No coverage floor, and `fail_ci_if_error: false` (ci.yml:108-114). ruff and mypy versions are unpinned. | The required checks are green by construction. That is how C1 shipped. | Drop `\|\| true` (or use `--severity-level high` plus an artifact upload); replace `safety check` with `pip-audit`. Add `ruff format --check` and `--cov-fail-under`. Re-enable mypy codes package by package, starting with `name-defined` and `attr-defined`. Pin tool versions. |
| 2 | **SECURITY.md has no reporting channel.** It says `**Email:** [Your security email]` (SECURITY.md:17) right after "do not report through public issues". It also advertises Bandit and Safety scanning (:59-60), which cannot fail. | Reporters have nowhere private to go. | Enable GitHub private vulnerability reporting or give a real address. Reword the scanning claims. |
| 3 | **pre-commit is documented but not configured.** CONTRIBUTING.md:66-68 and README.md:391 say `pre-commit install`, but there is no `.pre-commit-config.yaml` in the repo or its history. REVIEW_FEEDBACK.md:392 wrongly says "ALREADY CONFIGURED". | Once installed, the hook fails every commit. | Add a config (ruff, ruff-format, check-yaml) or remove the step. |
| 4 | **Contributor workflow references things that don't exist.** CONTRIBUTING.md:94, 103-104, 310 and 350 use a `develop` branch; the remote has only `main` and no tags. CI `pull_request` only targets `main` (ci.yml:6-7). Line length is 88 in CONTRIBUTING.md:140 but 100 in pyproject.toml:116. CONTRIBUTORS.md (CONTRIBUTING.md:380) doesn't exist. The required checks in RULESET_GUIDE.md:38-45 (`lint`, `type-check`, `test (3.10)`) don't match the job names (`Lint`, `Type Check`, `Test Python 3.10`, ci.yml:14, 34, 78). CODEOWNERS:5 and dependabot.yml:11, 27 use `@malavpatel` while the repo owner is `mlvpatel` (pyproject.toml:103). | Following the guide fails. Required checks would block forever, and code-owner review may never be satisfiable. | Align CONTRIBUTING with trunk-on-main (or create `develop` and add it to CI). Fix the check names and the owner handle. |
| 5 | **Dependency sources have drifted.** `langgraph>=1.0.10` (requirements.txt:35) vs `>=0.1.0` (pyproject.toml:53); pydantic-ai and llama-index floors differ too. requirements.txt also carries dev tools (:99-106) and weasyprint (:91), and Docker builds from it (Dockerfile:21-24). USER_MANUAL.md:47 and DEPLOYMENT.md:33 install via `-r requirements.txt`, which doesn't install the `agentds` CLI used at USER_MANUAL.md:166. `click` (cli.py:14, the console-script entrypoint) and `rq` (job_queue.py:201) are imported but not declared. Without rq, Redis silently falls back to memory (job_queue.py:284-299). duckdb, pydantic-ai, llama-index, mlflow, python-dotenv and rich are declared but never imported, and so is every `all` extra except docker. The CI cache key hashes requirements.txt (ci.yml:95) while installs come from pyproject. | Two sources of truth, a bloated prod image, and silent feature loss. | Make pyproject the single source; generate a lock or constraints file. Docker should `pip install .`. Declare click and rq, and prune unused deps. |
| 6 | **Unwired modules are presented as features.** validation.py (488 lines), cache_layer.py (332), core/apo.py (525) and integrations/ (557) have zero production callers. README and CHANGELOG sell "Input sanitization", "Redis-backed caching" and "Path traversal protection", and API_REFERENCE lists `PATH_TRAVERSAL` / `FILE_SIZE_LIMIT` errors that are never emitted. API inputs `data_source` and `output_destination` go unchecked (webhooks.py:174-179). | Misleading docs, dead weight, and false security assurance. | Wire `sanitize_path` / `validate_url` into the API, CLI and ArtifactStore, or mark these modules experimental and remove the README claims. |
| 7 | **Dead code.** Never called: `get_approval_message`, `validate_inputs`, `aexecute`, `acall_llm_structured` (base.py:494, 482, 411, 319); `get_agent` (agents/__init__.py:78); `run_apo_optimization` (optimization.py:486); `get_compiled_graph` (pipeline.py:583); `require_admin_key` (middleware.py:428). `RedisQueue._queue = Queue(...)` is created and never used (job_queue.py:204), so the compose `worker` service has nothing to run (docker-compose.yml:159-164). `config/pipeline_config.yaml` is never read (`get_pipeline_config` has no callers). In the UI, the HITL buttons are hidden forever, `job_id_state` is never set, `llm_preset` is unused, and Save Configuration has no handler (app.py:68, 151-171, 196-234). `prev_result.outputs.get(...)` results are discarded (api_wrapper.py:102-105). | Readers can't tell what is live. | Delete it, or wire it and test it. |
| 8 | **The DevOps feature flags can never be on.** `is_feature_enabled("github_actions"/"kubernetes_configs")` (devops.py:102, 105) only searches `features:` (config.py:296-311). Those flags live under `agent_features.devops` (feature_flags.yaml:133-138). | K8s and GHA generation is permanently off despite `enabled: true`. | Support dotted lookups (e.g. `agent_features.devops.github_actions`) or move the flags. |
| 9 | **Test quality is low relative to test count.** 112 of 182 tests (validation 51, exceptions 34, APO 27) cover unwired modules. pipeline.py, llm_gateway.py, cli.py, web/app.py and integrations have 0 tests, and no agent's `execute()` is tested. The 11 API integration tests are deselected by default (pyproject.toml:171). The ones that exist accept 500, or both 200 and 401 (test_api.py:197, 203, 209, 255, 264, 273), and one passes even if rate limiting never triggers (test_api.py:309-324). They also can't test what they configure, because `create_api(settings)` ignores `settings` for its middleware (webhooks.py:495-512 passes the classes, and the middleware calls `get_settings()`). Other problems: a tautology `assert settings.debug is True or settings.debug is False` (test_core.py:38) plus an `os.environ["DEBUG"]` leak (:35); `test_agent_chain` only checks `hasattr` (test_agents.py:324-342); 2s of sleeps not marked `slow`. | The suite gives false confidence; it missed C1–C4. | Assert exact codes, pass settings through to the middleware, and add offline agent `execute()` tests and pipeline tests with a mock gateway (CONTRIBUTING's own `test_load_csv` example is a good start). |
| 10 | **Releases are unsafe.** release.yml publishes on any `v*.*.*` tag with no tests and no tag-versus-version check. `version = "1.0.0"` is static (pyproject.toml:7) and also hardcoded at config.py:220, webhooks.py:151, 506, Dockerfile:33, the k8s labels and `__init__.py:22`. `skip-existing: true` turns an un-bumped tag into a silent PyPI no-op. The wheel ships only `agentds/` (pyproject.toml:111-112), so the config/*.yaml files are missing and every feature flag reads off. It uses `softprops/action-gh-release@v1` (release.yml:39). CHANGELOG has two `[1.0.0]` sections (docs/CHANGELOG.md:65, 138). | A broken or mislabelled release is likely, and installed behaviour differs from a checkout. | Run `needs: test` and a version-equals-tag check. Single-source the version (hatch dynamic, or read `__version__`). Package the default configs via `importlib.resources`. Fix the CHANGELOG. |
| 11 | **Docs and code disagree.** README.md:132-135 calls `pipeline.run("data.csv", task=...)`, which is a TypeError (the parameter is `task_description`). API_REFERENCE.md:313 and CHANGELOG.md:171 import `PipelineConfig`/`PipelinePhase` from `agentds`, which doesn't export them (`__init__.py:38-57`). The SSE `/api/pipeline/stream/{job_id}` (API_REFERENCE.md:458) doesn't exist. The documented error shape and codes (`message`/`details`, `JOB_NOT_FOUND` 404) differ from `ErrorResponse(error, detail)` and the 400 returned for an unknown agent (webhooks.py:100-105, 319-326). README.md:59 says "FastAPI generation" but the code generates Litestar (api_wrapper.py:213). The README anchors `DEPLOYMENT.md#aws/#gcp/#azure` are broken. "A/B testing support" exists only as unused `ab_*` settings. | Users copy these examples and they fail. | Fix the docs, or export the names. Add a CI job that runs a docs-snippet smoke test. |
| 12 | **Root hygiene.** REVIEW_FEEDBACK.md and REVIEW_SUMMARY.md are a point-in-time bot review (2026-02-12) and are already wrong (pre-commit "configured", "production-ready", "path traversal protection"). They sit at the root and ship in the sdist. There is a stale merged branch `copilot/review-repo-structure` and 8 unmerged dependabot branches. The CHANGELOG lives under docs/. | Noise, and stale authority. | Move the findings to issues and delete the files (or move them to `docs/reviews/`). Prune branches. Move the CHANGELOG to the root. |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| 1 | Enable `RUF100`. Remove the redundant `tests/*` ignores (S101 isn't selected; ARG001/002 are already global, pyproject.toml:142), the unused `# noqa: F401` (apo.py:500) and the `# noqa: I001` lines (web/api/__init__.py:9, test_api.py:9). | Fewer suppressions. |
| 2 | Import lazily in `agentds/__init__.py:30-36`. cli.py:16 imports `__version__`, which pulls in langgraph, litellm and every agent just for `--help`. | CLI startup time. |
| 3 | Rename `agentds.core.ValidationError` (core/__init__.py:38). | It shadows pydantic's `ValidationError`. |
| 4 | Remove the duplicate Services, which are defined in both deployment.yaml (:100, 202, 275) and service.yaml with different labels. Drop the obsolete compose `version:` (docker-compose.yml:7). | Avoids drift. |
| 5 | The Dockerfile builder uses `pip wheel --no-deps` (:24), so transitive deps get fetched in the runtime stage, which has no build tools. | The builder stage is pointless as written. |
| 6 | Drop the License classifier now that the license is a PEP 639 string (pyproject.toml:10, 31). | Redundant metadata. |
| 7 | Settings creates `outputs/ temp/ logs/ checkpoints/` in the current directory on every construction (config.py:270-280); `checkpoints/` isn't gitignored. | Tests and the CLI litter the tree. |

#### Convention Compliance Checklist
| Convention | Status | Notes |
|-----------|--------|-------|
| File placement | CONCERNS | Packages are well split (core/agents/web/workflows/integrations). The review files are at the root, the CHANGELOG is in docs/, and pipeline_config.yaml is never read. |
| Naming conventions | PASS | snake_case modules, PascalCase classes, consistent `*Agent` suffix. |
| Barrel exports (`__init__`/`__all__`) | CONCERNS | Good per package, but docs import `PipelineConfig`/`PipelinePhase` from the top level, which doesn't export them. |
| Import ordering | PASS | ruff `I` is clean, apart from 2 `noqa: I001`. |
| Type imports separated | N/A | No Python analogue in use; `from __future__ import annotations` is consistent. |
| Path aliases used | N/A | Absolute `agentds.` imports are used everywhere. |
| No dead code left | FAIL | Recommendations 6 and 7. |
| Structured logging | PASS | structlog `get_logger` everywhere; no `print()` in the package. |
| No lint suppressions | FAIL | The F821 per-file ignore masks C1; 13 mypy codes are off; 3 `type: ignore[abstract]`. |
| CLAUDE.md present | N/A | No CLAUDE.md convention here; CONTRIBUTING.md fills that role and is stale. |

#### Reuse Opportunities
- **JSON parsing of LLM replies.** Six agents hand-roll `re.search(r'\{.*\}', ...)` plus `json.loads` (data_cleaning.py:293, eda_copilot.py:277, data_loader.py:158, api_wrapper.py:184, automl.py:233, feature_engineer.py:220). `BaseAgent.call_llm_structured` (base.py:223) does exactly this with Pydantic validation, and no agent calls it.
- **Sync/async pairs in base.py.** `call_llm`/`acall_llm` and `call_llm_structured`/`acall_llm_structured` duplicate message building and parsing. Extract `_build_messages()` and `_parse_structured()`.
- **Two APO implementations.** core/apo.py (`APOOptimizer`, `BeamSearch`, `PromptHistory`, driven by `APOSettings`) and agents/optimization.py (`PromptTemplate`, its own cycle, `gradient_model="gpt-4o"` hardcoded at :517). They also have two on/off switches: the `agent_lightning_apo` flag and `APOSettings.enabled`. OptimizationAgent should use `create_apo_optimizer`.
- **Agent lookup and context setup.** cli.py:148-167, webhooks.py:318-345 and pipeline.py:155-163 each repeat the registry check, instantiation and `AgentContext` build. `get_agent()` already exists unused; add a `build_context()` factory.
- **API errors.** webhooks.py builds `Response(content=ErrorResponse(...).model_dump(), status_code=...)` 8 times. exceptions.py already has `JobNotFoundError`, `AgentNotFoundError` and `to_dict()`. Raise those and register one Litestar exception handler.
- **Path checks.** `sanitize_path(base_dir=...)` should guard `ArtifactStore._get_artifact_path` (artifact_store.py:105-110) and the API path inputs. Preflight's `startswith(("http","s3://","gs://"))` (pipeline.py:296) should use `validate_url`.
- **Generated requirements.** Two generators emit different requirements.txt files for the same generated app (api_wrapper.py:440 and devops.py:468). Keep one.
- **Version string.** Replace every hardcoded `"1.0.0"` with `agentds.__version__`.

#### Verdict Details
- File Organization: CONCERNS
- Naming Consistency: PASS
- Import Hygiene: PASS
- Code Reuse: CONCERNS
- Lint Compliance: CONCERNS (green only because of suppressions; 26 .py files unformatted)

#### Confidence notes
- **C2 is inferred, not run.** The pydantic-settings behaviour (list fields JSON-decoded before validators; nested `BaseSettings` built by `default_factory` not reading the parent's `env_file`) comes from my knowledge of the library. The project deps aren't installed, so I couldn't execute it.
- **C5 assumes the image isn't published elsewhere.** No workflow in the repo builds or pushes it; I can't see GHCR.
- **The `@malavpatel` handle** might be a valid second account; it differs from the `mlvpatel` used everywhere else.
- **mypy was run without third-party deps.** With the 13 overrides removed it reported only 3 errors, including C1. With real deps installed, the disabled codes probably hide more.
- **The format count** (26 of 42 .py files) is from ruff 0.16.9; CI's unpinned ruff could differ slightly.
- **I didn't verify everything.** C3 and C4 come from reading the code. I didn't check the old `MemorySaver` import path for `langgraph` 0.1.x, or whether hatchling rejects the PEP 639 license plus classifier combination.
- **Files I touched:** my `ruff check` may have written `.ruff_cache/` in the repo (gitignored). The `__pycache__/` and `.mypy_cache/` dirs that appeared at about 23:24 were not created by me; I ran mypy with a scratchpad cache. I made no other filesystem changes.

---

## 7. Production Readiness

**Verdict**: NEEDS REWORK

*Production Readiness Audit: Repo Health — main@d29bbd5*

**Perspective**: Production Readiness Engineer
**Date**: 2026-11-03
**Plan**: Repo Health — main@d29bbd5 (no plan or diff; the codebase was audited as it stands)

#### Plan Summary
What ships: a Gradio UI (`python -m agentds.web.app`, :7860) and a Litestar API (`python -m agentds.web.api.webhooks`, :8000), deployed by a Dockerfile, docker-compose and k8s manifests, with Redis, MLflow, an RQ worker and optional Ollama alongside. Both entrypoint commands and `python -m rq.cli worker` resolve to real modules. Several things break this in production:
- **Startup:** the web container crashes on start when no Logfire token is set.
- **Exposure:** neither the UI nor the API has working auth.
- **API restarts:** API pods restart-loop on LLM latency, and pipelines run inside the API's event loop.
- **Pipeline flow:** the default human-in-the-loop flow never runs and the UI fakes success. The job ID returned to API clients never exists in the queue.
- **k8s bring-up:** the manifests overwrite real secrets, cannot write to disk, use an unsupported RWX volume and reference an image that is never published.

#### Files Reviewed
| File | Production Role | Risk |
|------|----------------|------|
| docker/Dockerfile | Image build, CMD, healthcheck, user | High |
| docker/docker-compose.yml | Single-host stack (web, api, redis, mlflow, worker) | High |
| docker/k8s/deployment.yaml, service.yaml | Namespace, Secret, Deployments, PVCs, HPA, Ingress, NetworkPolicy | High |
| .github/workflows/ci.yml, release.yml | Test, build and release (no image build) | Med |
| agentds/web/app.py | Gradio UI entrypoint and pipeline trigger | High |
| agentds/web/api/webhooks.py, middleware.py | REST API, health, auth, rate limit | High |
| agentds/workflows/pipeline.py | LangGraph orchestration and job lifecycle | High |
| agentds/core/config.py, .env.example, config/*.yaml | Settings, flags, model routing | High |
| agentds/core/llm_gateway.py | LiteLLM calls, retries, fallback, health | High |
| agentds/core/job_queue.py, cache_layer.py | Redis/RQ job store, cache | High |
| agentds/core/artifact_store.py | Output persistence and metadata index | Med |
| agentds/core/logger.py | structlog + Logfire setup | High |
| agentds/agents/base.py, data_loader.py, cloud_deploy.py (+ automl, feature_engineer, devops) | Agent execution, file I/O, external calls | Med |
| docs/DEPLOYMENT.md | Operator runbook | Med |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | File(s) | Production Failure Mode | Fix |
|---|-------|---------|------------------------|-----|
| 1 | **Web container crash-loops at startup because of Logfire** | `core/logger.py:44-48,60-61`; `web/app.py:438`; `tests/conftest.py:20` | `logfire.configure(..., send_to_logfire=settings.environment != "test")` is True in every deployment, because ENVIRONMENT is never set and defaults to "development". No `LOGFIRE_TOKEN` is set in .env.example, compose, k8s or the docs. Logfire then goes down the `initialize_project` path and raises `LogfireConfigError`, but only `ImportError` is caught. The Dockerfile CMD, compose `web`, k8s `agentds-web` and `agentds web` all exit before binding :7860. CI hides this because conftest sets ENVIRONMENT=test. | Use `send_to_logfire="if-token-present"` and catch `Exception` in `configure_logfire`. Add a CI smoke test that boots the container. |
| 2 | **Neither public surface has effective auth** | `web/api/middleware.py:217-220`; `core/config.py:241,261-267`; `.env.example:134`; `docker-compose.yml:59-65`; `k8s/deployment.yaml:237-251,363-379`; `web/app.py:447-451` | **API fails open:** with no keys configured, `validate()` returns True for any non-empty `X-API-Key`. Neither compose nor k8s passes `API_KEYS`. **Documented format crashes:** `API_KEYS=key1,key2,key3` as documented makes `Settings()` crash at startup, because pydantic-settings JSON-decodes `list[str]` before the `mode="before"` validator runs. Local quick-start after `cp .env.example .env` crashes the same way. **UI has no auth:** Gradio launches without `auth=`, and the Ingress publishes both `/` and `/api`. **Impact:** anyone on the internet can spend your LLM keys and point `data_source` at local files (e.g. `/var/run/secrets/...`), internal URLs or DB DSNs (`data_loader.py:244-330`). Column names and stats come back in `/api/agent/run` responses. | Fail closed when no keys are set outside development. Use `Annotated[list[str], NoDecode]` or a string field, and inject the keys from a Secret. Put auth in front of Gradio (ingress OIDC or `auth=`). Allow-list `data_source` schemes and paths. |
| 3 | **`/api/health` makes a blocking, retried, paid LLM call and is used for both liveness and readiness** | `web/api/webhooks.py:127-147`; `core/llm_gateway.py:183-187,457-475`; `k8s/deployment.yaml:259-270` | Each probe runs a synchronous `complete()` with 3 tenacity attempts, `timeout=120` and chain `[model, model]`, inside `async def`, so it blocks the event loop. The probes set no `timeoutSeconds`, so the 1s default applies. Any LLM latency above 1s, or an outage, makes the API pods restart-loop and takes the job and status endpoints down with them. Every probe (2 pods × 8/min) and every anonymous hit through the ingress costs a completion. | Liveness should check only the process (a trivial 200). Readiness should be a Redis ping with a timeout. Run any LLM check in the background, cache it, and keep it out of probes. |
| 4 | **Pipelines run inside async API handlers, and the worker has nothing to consume** | `web/api/webhooks.py:174-179,285-289,348`; `core/job_queue.py:201-218`; `docker-compose.yml:159-179` | `pipeline.run`, `resume` and `agent.run` execute synchronously on the uvicorn event loop. An RQ `Queue` is created but nothing is ever enqueued, so the `worker` service idles. A `human_in_loop=false` run freezes the API for the whole pipeline (LLM calls plus AutoML). Probes then fail, the kubelet kills the pod mid-run, and the job stays QUEUED forever. | Enqueue to RQ and return 202 with the job ID; the worker executes and persists status. At minimum, offload with `sync_to_thread`. |
| 5 | **The core end-to-end flow cannot complete or be observed** | `workflows/pipeline.py:526,559-567,604-624`; `web/api/webhooks.py:42,211`; `web/app.py:111-114,304-323`; `core/job_queue.py:48,322` | **(a) The default flow never runs.** With `human_in_loop=True`, the default in both API and UI, `run()` never invokes the graph and returns "started". There is no `interrupt_before` and nothing is saved in `MemorySaver`, so `resume()` restarts at preflight with no `data_source` and fails. **(b) The UI fakes success.** Gradio steps through `pipeline._agents` with `time.sleep(0.5)` and then shows "Pipeline completed successfully!". `job_id_state` is never set and the approval row is `visible=False`, so the approve and stop buttons do nothing. **(c) Job IDs never match.** `create_job()` mints its own UUID, so the job ID returned by `/api/pipeline/start` never exists in the queue. `/api/pipeline/status/{id}` and `/cancel` always return 404/400. | Compile the graph with `interrupt_before=["human_review"]` and a Redis or Postgres checkpointer, and drive the UI from job state. Pass the job ID into `create_job`. Delete the simulated progress. |
| 6 | **The documented k8s rollout overwrites real API keys with placeholders** | `k8s/deployment.yaml:35-43`; `docs/DEPLOYMENT.md:111-117` | The manifest ships `Secret agentds-secrets` with `stringData: OPENAI_API_KEY: "replace-with-actual-key"`. The runbook says to create the secret and then `kubectl apply -f docker/k8s/`. The apply replaces the real keys, so every LLM call returns 401. It happens again on every re-apply, such as an image bump. | Remove the Secret object from the manifest, or use ExternalSecret/SealedSecret. |
| 7 | **The k8s manifests cannot run the app as shipped** | `docker/Dockerfile:37`; `k8s/deployment.yaml:145,195-197,233,288-299`; `agents/data_loader.py:106-107`; `.github/workflows/release.yml:27-28` | **(a) UID mismatch.** `useradd -r` creates UID 999 (Debian system range), but the pod forces `runAsUser: 1000` with no `fsGroup`. `/app/temp` and `/app/logs` (owned by 999, mode 0755) and the root-owned PVC are unwritable. DataLoaderAgent's `write_parquet` to `temp_dir` and every artifact save fail, so every pipeline fails. **(b) RWX volume.** The outputs PVC asks for `ReadWriteMany`, which the default storage classes on EKS, GKE and AKS (the targets DEPLOYMENT.md names) do not provide. The PVC stays Pending and so do the web pods. **(c) Image never published.** No workflow builds or pushes `ghcr.io/mlvpatel/agentds:latest`; release builds only the sdist and wheel. Pods go to ImagePullBackOff. | Create the user with `useradd -u 1000` (or drop `runAsUser`) and add `fsGroup: 1000`. Use RWO volumes or object storage. Add a docker build/push job with immutable tags. |
| 8 | **Agent models need provider keys the deploy configs never inject, and failed agents are recorded as success** | `config/llm_config.yaml:34-35,46-47`; `k8s/deployment.yaml:155-164,244-248`; `core/llm_gateway.py:305-317`; `agents/data_loader.py:154`; `workflows/pipeline.py:326-337,570-575` | DataLoaderAgent is mapped to `groq/llama-4-scout`, but no GROQ key reaches either k8s pod. EDA is mapped to `google/gemini-2.5-flash`, and `google/` is not a LiteLLM provider prefix. The k8s API pod gets only `OPENAI_API_KEY`. Auth and BadRequest errors are not in the fallback's except clause, so they raise immediately. The pipeline ignores `AgentResult.status == FAILED`. On k8s, the first agent therefore fails on every run, downstream agents fail in turn, and the job is marked COMPLETED with empty outputs. | At startup, check that every mapped model has its provider key and fail readiness if not. Route failed agent results to the error node. |
| 9 | **Redis is published on the host without auth, and its contents are read with `pickle.loads`** | `docker-compose.yml:84-91`; `core/job_queue.py:224-227,251-252` | Compose publishes `"6379:6379"` with no `requirepass` (MLflow :5000 and Ollama :11434 are also published without auth). Docker-published ports bypass ufw. Anyone who can reach the host can write a crafted value into `agentds:jobs`. The next Jobs-tab refresh or `GET /api/jobs` then runs it inside the web and API containers, which hold all the LLM keys. | Don't publish Redis (or bind it to 127.0.0.1). Add `requirepass`/ACL. Store jobs as JSON (`model_dump_json`). |
| 10 | **Outputs and run state from API-triggered pipelines are lost on every pod restart** | `k8s/deployment.yaml:215-270`; `agents/data_loader.py:106`, `automl.py:146-148`, `feature_engineer.py:140-145`, `devops.py:109`; `workflows/pipeline.py:152`; `web/api/webhooks.py:112` | The API Deployment mounts no volume. Every agent writes intermediates and models to `temp_dir`, which is never a volume in compose or k8s. `MemorySaver` and `_active_pipelines` live in the process. Every rollout, HPA scale-down or OOM kill deletes models and reports produced through the API (the n8n path) and orphans paused runs. | Give the API and worker shared storage (or S3) for outputs and temp files, and use a persistent checkpointer. |

#### Recommendations
| # | Issue | Why | Fix |
|---|-------|-----|-----|
| 1 | The LLM fallback chain modifies the shared config list and misses common errors (`llm_gateway.py:178-181,229-230,305`) | `FallbackChain(models=models)` points at the same list as `self.fallback_chains["default"]`, and `insert(0, model)` adds a copy on every call for as long as the pipeline's gateway lives. After N calls, a slow primary is retried N×120s before failover, and rate-limited calls hammer the provider in a tight loop. 5xx, connection and auth errors never fail over. | Copy the list (`list(models)`) and dedupe it. Also catch `APIConnectionError`, `InternalServerError` and `ServiceUnavailableError`. Give each call a total time budget. |
| 2 | Redis clients have no timeouts and fall back silently (`job_queue.py:203,284-299`; `cache_layer.py:131`; `config.py:110` unused) | A partitioned Redis hangs request threads indefinitely. A Redis blip at pod start leaves that pod on a private `InMemoryQueue` forever: its jobs are invisible to other pods and lost on restart. Redis loss after startup shows up as raw 500s. | Pass `socket_timeout`, `socket_connect_timeout` and `health_check_interval`. In production, fail readiness instead of falling back. Add retry with backoff. |
| 3 | The Redis job store can lose everything at once and breaks across versions (`job_queue.py:205,251-252`; `docker-compose.yml:91`; `deployment.yaml:47-95`) | All jobs sit in one hash under `allkeys-lru`, so eviction can drop the entire history. maxmemory 512mb equals the 512Mi limit, so an AOF rewrite can trigger an OOM kill. Jobs are stored as pickled `Job` objects: any class change makes old entries unreadable, and one bad entry fails the whole listing. A Deployment with an RWO PVC and the default RollingUpdate can get stuck on Multi-Attach. | Store one JSON key per job with a TTL. Use `noeviction` or `volatile-lru`, with maxmemory around 70% of the limit. Run Redis as a StatefulSet or with `strategy: Recreate`. |
| 4 | Gradio runs on 2–10 replicas with no session affinity (`deployment.yaml:134,302-321,346-379`) | Gradio's queue join and its SSE stream must reach the same replica. Round-robin breaks streaming, and `_active_pipelines` (`app.py:26`) exists per pod. | Add nginx cookie affinity annotations, or run one UI replica and scale workers instead. |
| 5 | The artifact index is written in place and the last writer wins (`artifact_store.py:81,96-103`) | Each instance loads `.artifacts.json` once and rewrites the whole file on every save, on a shared PVC. Concurrent pipelines and pods drop each other's entries. A crash mid-write leaves truncated JSON; the next load returns `{}` and the next save wipes the index. DevOpsAgent reads artifacts back from this index (`devops.py:115`). | Write to a temp file and `os.replace`, keep metadata per job, or move it to Redis/DB. Add a file lock. |
| 6 | No graceful shutdown or cleanup of interrupted jobs (`Dockerfile:77`; k8s specs) | Gradio runs as PID 1 and installs no SIGTERM handler (its uvicorn runs in a thread), so SIGTERM is ignored until SIGKILL. There is no grace-period tuning or `preStop`, and nothing marks jobs on exit or at startup. Every deploy hard-kills running pipelines mid-write and leaves jobs RUNNING/QUEUED forever. | Add `tini`/`--init`. On SIGTERM, drain and mark jobs `interrupted`. Sweep stale RUNNING jobs at startup. Size the grace period to agent timeouts. |
| 7 | The container build differs from the tested environment and is not reproducible (`Dockerfile:24,48`; `requirements.txt`; `ci.yml`) | `pip wheel --no-deps` followed by `pip install /wheels/*` drops the extras (`polars[all]`, `smart-open[all]`, `uvicorn[standard]`) and resolves transitive deps from PyPI at build time. So `.xlsx` upload, offered at `app.py:90`, lacks fastexcel in the image. Every requirement is `>=`, dev tools are shipped, and CI tests the pyproject dependency set without ever building the image. | Use one lock file for CI and Docker. Build wheels with deps and install with `--no-index`. Split out dev deps. Add a CI image build plus smoke test. |
| 8 | LLM output decides how data is loaded (`data_loader.py:146-160,307-308`) | The model's JSON chooses `source_type`, `format` and `encoding`, and supplies the SQL `query`. Valid inputs fail unpredictably, and a task description can steer arbitrary SQL against a supplied DSN. | Detect the source deterministically. Use the LLM only for advisory notes, and never take SQL from it. |
| 9 | Health checks and logs report "fine" when things are broken (`webhooks.py:132-135,149-154,517-521`; `app.py:369-376`; `logger.py:171`; `DEPLOYMENT.md:406-411`) | Health returns 200 with `job_queue: True` hard-coded. The API never calls `setup_logging`, so its logs are not JSON and it has no Logfire. No log file is ever written, so the Logs tab is always empty. `/metrics` is documented but not served. MLflow is deployed but never imported. | Set up structured logging in both entrypoints. Report real dependency status in readiness. Emit start/agent/finish/fail events tagged with the job ID. Add a metrics endpoint or remove the docs. |
| 10 | In-process state and temp files grow without bound (`webhooks.py:112,182`; `app.py:26,290`; agents' `temp_dir` writes) | `_active_pipelines` is never evicted, and each entry holds the agents, the gateway and a Redis pool. Per-job parquet and pickle files are never cleaned, and no `ephemeral-storage` limits are set. Expect OOM kills and node-disk evictions. | Evict entries by TTL. Use a per-job temp dir removed in `finally`. Set storage requests and limits. |
| 11 | CloudDeployAgent cannot work in a container, yet it is in the default "deploy" phase (`cloud_deploy.py:109,163-168,190-211`) | Its default `docker_local` platform needs a Docker daemon that no container has. It binds host port 8000 and probes `localhost:8000/health`, which inside the API container is the API itself. It never cleans up containers. So the default build+deploy phases always end in a failed agent. | In server mode, default to generating instructions only, and make deployment an explicit opt-in. |
| 12 | Auth and rate limiting need hardening (`webhooks.py:499-512`; `middleware.py:76-81,171,204-208`) | Auth runs before the rate limiter, so 401s are never throttled. Behind the ingress, every client appears as the ingress pod's IP and shares one bucket. Limits are per pod and multiply with HPA. `?api_key=` puts keys in access logs. CORS allows `*`. | Rate-limit first, keyed on X-Forwarded-For from a trusted proxy. Use a Redis-backed limiter, accept keys only in headers, and list explicit origins. |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| 1 | Override the healthcheck for compose `worker` | It inherits `curl :7860/` (`Dockerfile:73-74`) and is permanently unhealthy |
| 2 | Pin `redis:7-alpine`, `ollama/ollama:latest` and the app's `:latest` to versions or digests | Makes rollback deterministic |
| 3 | Drop the duplicate Services in `k8s/service.yaml` | Same names as in deployment.yaml but different labels; they will drift |
| 4 | Use the `LLM_` prefix for `DEFAULT_TEMPERATURE`, `MAX_RETRIES` and `REQUEST_TIMEOUT` (`.env.example:144-150`, ConfigMap), and set `ENVIRONMENT=production` | Today these values are silently ignored, and prod runs as "development" |
| 5 | Add a `.dockerignore` | Keeps `.env`, `outputs/` and `.git` out of the build context |
| 6 | Add a startupProbe for web and a PodDisruptionBudget | Imports are heavy; nothing protects pods during voluntary disruptions |
| 7 | Wire up or remove the Configuration tab's key fields and Save button (`app.py:196-232`) | Keys typed there are silently discarded |
| 8 | Align the OpenAPI path (Litestar default `/schema`) with `EXEMPT_PATHS` and the ingress | The docs currently can't be reached |
| 9 | Add a timeout to `smtplib.SMTP(...)` in `async def` (`notifications.py:227-230`) | Will block the loop once notifications are wired in |
| 10 | Set `GRADIO_ANALYTICS_ENABLED=False` | Stops prod pods phoning home |

#### Failure Mode Analysis
| Scenario | What Fails | User Impact | Recovery | Plan Handles It? |
|----------|-----------|-------------|----------|-------------------|
| No `LOGFIRE_TOKEN` (the default) | `configure_logfire` raises at web startup | UI never comes up | Crash loop | No |
| LLM provider slow (>1s) | `/api/health` probe times out | API pods restart repeatedly | None | No |
| LLM 5xx, connection error or missing key | No failover; agent FAILED; pipeline carries on | Job shows "completed" with no outputs | None | No |
| Primary LLM times out late in a run | Fallback chain retries the duplicated primary N×120s | Pipeline hangs for a long time | Eventually | No |
| Redis down at pod start | Silent in-memory queue per pod | Jobs invisible across pods, lost on restart | Pod restart after Redis returns | Partial |
| Redis down or partitioned later | Unhandled errors, or hangs with no socket timeout | 500s or frozen requests | Manual | No |
| Redis memory pressure | LRU evicts `agentds:jobs`, or OOM kill | All job history gone | None | No |
| Pod SIGTERM mid-pipeline | SIGKILL after grace; artifacts half-written | Job stuck QUEUED/RUNNING | None | No |
| Crash during `.artifacts.json` write | Truncated index → `{}` → wiped on next save | Artifact index lost (files orphaned) | Manual rebuild | No |
| Concurrent saves from 2 pods or pipelines | Last writer wins on the index | Artifacts missing from listings | None | No |
| Gradio request lands on another replica | Queue/SSE mismatch | Stream errors, lost progress | User retries | No |
| Default HITL run via API or UI | Graph never invoked; UI shows fake success | Nothing is produced | None | No |
| Client polls `/api/pipeline/status/{job_id}` | Job ID mismatch → 404 | n8n/clients can't track jobs | None | No |
| `kubectl apply -f docker/k8s/` per runbook | Secret replaced by placeholders; RWX PVC Pending; image missing | Deploy is down or every LLM call 401s | Manual | No |
| API pod rescheduled | Outputs on ephemeral disk lost | Models and reports gone | None | No |
| Anonymous internet request to `/api/agent/run` | Accepted (fail-open auth) | Cost abuse, local file read, SSRF | None | No |
| Docker daemon absent | CloudDeployAgent fails | Deploy phase always fails | Agent marks failed | Partial |

#### Environment Checklist
| Concern | Dev | Prod | Gap? |
|---------|-----|------|------|
| File paths | Relative `outputs/`, `temp/`, `logs/` under the repo | `/app/temp` and `/app/logs` on ephemeral container disk; outputs PVC on web only, nothing on API | Yes |
| Network | `localhost` defaults for Redis, MLflow, Ollama (`config.py:58,108,123`) | Compose/k8s override `REDIS_URL` only; 0.0.0.0 binds with no auth; Redis, MLflow and Ollama published on the host | Yes |
| Permissions | Runs as the developer's user | Image UID 999 vs k8s `runAsUser: 1000`, no `fsGroup`; PVC root-owned | Yes |
| Config | `.env` with the full set of keys | No `ENVIRONMENT`, `API_KEYS`, `LOGFIRE_TOKEN` or GROQ/GEMINI keys; placeholder Secret; the documented `API_KEYS` format crashes Settings | Yes |
| Build output | `pip install -r requirements.txt` (extras installed) | `--no-deps` wheels drop extras; image never built in CI or published; nothing pinned | Yes |
| Secrets | `.env` file | k8s Secret placeholder overwrites real keys; API pod gets only the OpenAI key | Yes |
| Process lifecycle | Ctrl-C (SIGINT) works | Gradio as PID 1 ignores SIGTERM until SIGKILL; RQ worker is idle | Yes |

#### Verdict Details
- E2E Flow: CONCERNS. The default HITL path never runs, the UI fakes success, job IDs never match, and the k8s first agent fails for lack of a key.
- Error Recovery: CONCERNS. The health probe causes restart loops, fallback misses common errors, Redis has no timeouts, and there is no graceful shutdown or job cleanup.
- Data Integrity: CONCERNS. The artifact index is non-atomic and last-writer-wins; Redis uses LRU eviction plus pickle; API outputs are ephemeral.
- Environment Parity: CONCERNS. Logfire crashes only outside tests, the Docker build drops extras, UIDs don't match, and env vars go missing in compose and k8s.
- Observability: CONCERNS. Health always returns 200, the API has no structured logging, the logs tab is empty, `/metrics` is documented but absent, and MLflow is unused.

Checklist items with no direct analogue here:
- **Sleep/wake:** not applicable to a server.
- **DB schema migrations:** there is no database. The closest analogue is the pickled `Job` objects in Redis (Recommendation 3).
- **Feature flags for rollout:** `feature_flags.yaml` exists, but only `redis_queue`, `redis_cache`, `agent_lightning_apo` and two `devops` keys are ever read. The devops keys sit under `agent_features`, so they always read as False.
- **Bundle assets (the Tauri item):** the Dockerfile copies `agentds/` and `config/`, which is all the runtime needs.

#### Confidence notes
- **C1 (Logfire):** I confirmed against current logfire-sdk source that `send_to_logfire=True` with no token goes to `initialize_project` → `UserTokenCollection().get_token()` and raises `LogfireConfigError`, and that `configure()` calls `initialize()` synchronously. I did not run it, and older logfire 2.x may word the error differently, though the outcome should be the same. Setting `LOGFIRE_TOKEN` would avoid it.
- **C2 (`API_KEYS` crash):** this relies on pydantic-settings JSON-decoding complex fields from env and dotenv sources before validators run. That is the documented default, but it is not exercised in this repo.
- **C7a (UID 999):** assumes Debian `useradd -r` gives the first system user UID 999. **C7b (RWX):** depends on the cluster's storage class.
- **C8 (model IDs):** that `google/` is not a LiteLLM provider prefix, and that `groq/llama-4-scout` / `deepseek/deepseek-v3` are not valid upstream model IDs, comes from knowledge rather than a live call. The missing GROQ key in the k8s manifests is certain from the files.
- **Recommendation 1 (exception classes):** that LiteLLM's `APIConnectionError`, `InternalServerError` and `ServiceUnavailableError` are not subclasses of `litellm.APIError` is from knowledge of LiteLLM's hierarchy, not checked against an installed copy. The list-mutation bug is certain from the code.
- **Recommendation 4 (Gradio stickiness)** follows Gradio's own deployment guidance; I did not reproduce it.
- **Recommendation 6 (PID 1 ignoring SIGTERM)** is standard Linux PID-namespace behaviour. That Gradio installs no SIGTERM handler is from knowledge of Gradio's launch path.
- **Recommendation 7 (fastexcel):** assumes polars ≥1.x defaults to the `calamine` engine and that nothing else in the image pulls in fastexcel.
- **C9** is only exploitable when the compose host's port 6379 can be reached from outside.

---

## 8. DX Engineer

**Verdict**: NEEDS REWORK

*DX Audit: Repo Health — main@d29bbd5*

**Perspective**: Developer Experience Engineer
**Date**: 2026-11-03
**Repo Health — main@d29bbd5**: `/home/user/AgentDS` (no plan or diff, so I audited the codebase as it stands)

#### Summary — Repo Health main@d29bbd5
A developer who follows the README or docs/ hits a hard failure within the first five minutes, by any route: the Python snippet, `cp .env.example .env`, the CLI, the REST API or contributor setup. On top of that, the CLI, web UI and API all report success when nothing ran or when agents failed. Ruff passes (`ruff check .` gives "All checks passed!"). I did not run the tests or mypy because the project's dependencies are not installed.

#### Files Reviewed
| File | DX Role | Risk |
|------|---------|------|
| README.md, docs/USER_MANUAL.md, docs/API_REFERENCE.md | First-run docs, CLI, Python and REST reference | High |
| docs/TROUBLESHOOTING.md, LLM_PROVIDERS.md, N8N_GUIDE.md, APO_GUIDE.md, CHANGELOG.md, DEPLOYMENT.md | Task guides | High |
| CONTRIBUTING.md, .github/workflows/ci.yml | Contributor setup | High |
| .env.example, config/*.yaml | Configuration templates | High |
| agentds/cli.py | CLI entry point (`agentds`) | High |
| agentds/__init__.py | Public Python API | High |
| agentds/workflows/pipeline.py | `AgentDSPipeline.run/resume` | High |
| agentds/core/config.py, core/logger.py | Settings loading, logging, Logfire | High |
| agentds/core/exceptions.py, core/llm_gateway.py | Error surface, model selection | Medium |
| agentds/web/api/webhooks.py, middleware.py | REST contract | High |
| agentds/web/app.py | Gradio UI | Medium |
| agentds/agents/*.py, core/job_queue.py, core/artifact_store.py | Agent inputs, job IDs, output paths | Medium |
| docker/docker-compose.yml, docker/Dockerfile | Docker quick start | Medium |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | Impact on Developer | Fix |
|---|-------|-------------------|-----|
| 1 | **The Python quick-start snippets fail.** README.md:129-135 calls `pipeline.run("data.csv", task="Predict churn")`, but the signature is `run(self, data_source, task_description, output_destination=None, job_id=None)` (pipeline.py:503-509). API_REFERENCE.md:313 and CHANGELOG.md:171 use `from agentds import AgentDSPipeline, PipelineConfig, PipelinePhase`, but `agentds/__init__.py:30-57` exports neither `PipelineConfig` nor `PipelinePhase`. | The README snippet raises `TypeError: unexpected keyword argument 'task'`; the API reference and migration-guide snippets raise `ImportError`. | Export `PipelineConfig`, `PipelinePhase`, `AgentContext` and `AgentAction` from `agentds`. Accept `task=` as an alias or fix the README. Add a smoke test that runs the README snippets. |
| 2 | **Every surface reports false success.** cli.py:111 prints `[OK] Pipeline completed successfully!` unconditionally. With the default HITL setting (the README command at README.md:110-113 has no `--no-hitl`), pipeline.py:559-567 never invokes the graph (`final_state = initial_state`). With `--no-hitl`, agent exceptions are swallowed into `AgentResult` (base.py:467-474, plus each agent's own try/except), and preflight errors go into `state["error"]` (pipeline.py:226), which the CLI never checks. The web UI runs a fake loop (app.py:317-318: "Simulate agent execution", `time.sleep(0.5)`) and then says "Pipeline completed successfully!" (app.py:322). Its HITL buttons always reply "No active pipeline" because `job_id_state` (app.py:67) is never an output (app.py:379-383). The API returns "Pipeline started successfully" (webhooks.py:187-192). | Users believe models were trained. The exit code is 0, so scripts and CI cannot detect failure. The quick-start command does nothing at all. | CLI: default to no HITL (it has no approval loop) or add interactive prompts. Check `result["state"]["error"]` and failed agent statuses and exit 1 on either. Propagate agent failures into the pipeline state. Remove the simulated progress from the web UI. |
| 3 | **`cp .env.example .env` breaks settings loading.** .env.example:134 has `API_KEYS=key1,key2,key3`, and the target field is `api_keys: list[str]` (config.py:241). pydantic-settings tries to parse complex fields as JSON before the `mode="before"` CSV validator (config.py:261-267) ever runs. | Step 2 of the quick start makes every command fail with `SettingsError: error parsing value for field "api_keys"`. If the value did parse, the API would require the guessable keys `key1`, `key2` and `key3`. | Comment the line out (`# API_KEYS=`). Annotate the field with `NoDecode` or accept a `str` and split it. Leave placeholder values empty instead of `sk-...`. |
| 4 | **Logfire can abort every CLI command on a fresh machine.** The group callback calls `setup_logging` (cli.py:33-35). Click runs that callback even for `agentds run --help`. `setup_logging` calls `configure_logfire`, which calls `logfire.configure(send_to_logfire=settings.environment != "test")` (logger.py:44-48). `environment` defaults to `"development"` (config.py:222) and .env.example sets neither ENVIRONMENT nor LOGFIRE_TOKEN. Only `ImportError` is caught (logger.py:60). The tests avoid this only because conftest.py:20 sets `ENVIRONMENT=test`. The Docker web image goes through the same path via `launch_app` (app.py:438). | `agentds status`, the documented diagnostic, fails with "You are not authenticated. Please run `logfire auth`". That pushes users to sign up for a SaaS product the docs never mention. | Use `send_to_logfire="if-token-present"`, catch `Exception` and log a warning, and document LOGFIRE_TOKEN. |
| 5 | **The documented way to choose a model does nothing, so non-OpenAI users fail preflight.** The docs say to set `default_model` or `presets` in llm_config.yaml (USER_MANUAL.md:246-250; TROUBLESHOOTING.md:441-449 "run without any API keys"; LLM_PROVIDERS.md:358-366). The gateway reads only `agent_llm_mapping` and `fallback_chains` (llm_gateway.py:136-140). Preflight tests `settings.llm.default_model`, which is `openai/gpt-4o-mini` (config.py:63, llm_gateway.py:467, pipeline.py:249-253). The variable that actually controls it, `LLM_DEFAULT_MODEL`, appears in no doc and not in .env.example. | Ollama-only and Anthropic-only users, whom USER_MANUAL.md:31 says are supported, get "Pre-flight checks failed: LLM connection failed" and no way to fix it from the docs. | Read `default_model` and presets from the YAML, or document `LLM_DEFAULT_MODEL` in .env.example and the guides. Put the model name and the remedy in the preflight error. |
| 6 | **The REST "start, then poll" workflow cannot work.** `POST /api/pipeline/start` returns job_id X (webhooks.py:172-192). `pipeline.run` then creates the Job without that ID (pipeline.py:522-534; `Job.id` is a fresh UUID, job_queue.py:48), in the pipeline's own JobQueue (pipeline.py:144). So `/pipeline/status/X` always returns 404 (webhooks.py:211-220) and `/pipeline/cancel/X` always returns 400 (webhooks.py:239-249). The SSE endpoint `/api/pipeline/stream/{job_id}` (API_REFERENCE.md:460-492) and `/api/docs` (API_REFERENCE.md:25) do not exist; Litestar serves OpenAPI at its default `/schema`. | API clients, and the n8n polling workflow in N8N_GUIDE.md:113-150, never see a job complete. The CLI's printed "Job ID" (cli.py:112) does not match what `agentds jobs` lists (cli.py:285). | Pass the job_id into `create_job` and share one JobQueue. Implement the SSE endpoint or remove it from the docs. Set `OpenAPIConfig(path="/api/docs")`. |
| 7 | **`agentds agent <Name> <file>` fails for 8 of 10 agents.** The help text lists all 10 as runnable (cli.py:128-138), but the CLI only sets `extra["data_source"]` (cli.py:160-167). Only DataLoaderAgent reads that key (data_loader.py:92). The others read `previous_results` (e.g. automl.py:114-116, data_cleaning.py:97-99) or `extra["reference_data"]` (drift_monitor.py:97-107). | The documented example `agentds agent AutoMLAgent ./outputs/clean_data.parquet` (USER_MANUAL.md:209-210) always fails with "No input from FeatureEngineerAgent", which is not actionable from the CLI. | Map DATA_SOURCE to each agent's expected input (e.g. synthesize a previous result containing `data_path`) and add `--target`. Otherwise restrict the command and say so in its help. |
| 8 | **Contributor setup per CONTRIBUTING fails.** It says to run `pre-commit install` (CONTRIBUTING.md:66-68, README.md:391), but the repo has no `.pre-commit-config.yaml`. It also says to use a `develop` branch (`git checkout develop`, rebase onto `upstream/develop`, PRs against `develop`; CONTRIBUTING.md:93-108, 307-311, 350). `git ls-remote` shows only `main`, and CI runs for PRs to `main` only (ci.yml:4-7). | After `pre-commit install`, every `git commit` aborts with "No .pre-commit-config.yaml file was found". `git checkout develop` fails, and a PR against `develop` would get no CI. | Add a pre-commit config (ruff and mypy) or drop the step. Rewrite the branching section around `main`. |
| 9 | **The USER_MANUAL and DEPLOYMENT install path never installs the CLI.** USER_MANUAL.md:47 and DEPLOYMENT.md:33 use `pip install -r requirements.txt` with no `-e .`, yet USER_MANUAL.md:161-185 then uses `agentds ...`. The README uses `pip install -e .`. | `agentds: command not found` for anyone who followed the manual. | Use a single install command everywhere: `pip install -e ".[dev]"` (or `.[all]`). |

#### Recommendations
| # | Issue | Why | Fix |
|---|-------|-----|-----|
| 1 | `-o/--output` is ignored. No agent reads `output_destination`; artifacts go to `settings.temp_dir` and `settings.output_dir` relative to the working directory (data_loader.py:106, artifact_store.py:78). The CLI still prints `Outputs: {output}` (cli.py:113). | Users look in the directory they named and find it empty. | Thread the output directory into `ArtifactStore` and the agent context, and print the real artifact paths. |
| 2 | Much of the documented configuration does nothing. `pipeline_config.yaml` is never loaded (`get_pipeline_config` has no callers), yet USER_MANUAL.md:405-417 and :500 and TROUBLESHOOTING.md:455-461 tell users to edit it. LLM presets are unused, and the web UI's preset dropdown (app.py:115-117) is ignored. `HUMAN_IN_LOOP` and `AUTO_APPROVE_LOW_RISK` are loaded (config.py:247-248) but `PipelineConfig` never uses them (pipeline.py:52-53). In .env.example, `DEFAULT_TEMPERATURE` through `REQUEST_TIMEOUT` (lines 144-150) lack the `LLM_` prefix their settings class requires (config.py:25). The storage and database variables (.env.example:80-117) are read nowhere. USER_MANUAL.md:229 says `TOGETHER_API_KEY` but the code reads `TOGETHERAI_API_KEY` (config.py:36). The web UI's API-key fields and "Save Configuration" button (app.py:197-232) have no handler, yet USER_MANUAL.md:136 says "Keys are stored securely". | The config file is the documentation. Settings that do nothing waste debugging time and hide the settings that do work. | Wire these settings up or delete them. Warn at startup about unknown keys. Add `agentds config --sources`. |
| 3 | Nested settings do not read `.env`. Only the outer `Settings` class sets `env_file=".env"` (config.py:211-216). `LLMSettings` and `RedisSettings` are built with `default_factory` and have no `env_file` (config.py:24-28, 103-106, 251-252). | API keys in `.env` reach `settings.llm` only if something else already exported them into the process environment, which depends on the setup. | Call `load_dotenv()` at CLI and app startup, or use `env_nested_delimiter` or a shared `env_file`. |
| 4 | `agentds status` cannot fail and over-reports. Ollama is always added to the provider list (config.py:95-96), so the "No LLM providers configured" warning (cli.py:199-200, documented at USER_MANUAL.md:470) can never appear. Placeholder values like `sk-...` count as configured. Nothing is tested live, and the exit code is always 0. | The documented diagnostic tool gives false confidence and can't be used in scripts. | Add `--check`, which calls `validate_connection` for each provider. Treat placeholders as missing. Exit non-zero when no provider is usable. Add `--json`. |
| 5 | The CLI does not validate its arguments. Free-text `--phases` goes to `PipelinePhase(p)` outside the try block and produces a traceback (cli.py:70-71, 95). `click.Path(exists=True)` (cli.py:67, 122) rejects the s3/gs/http sources that preflight accepts (pipeline.py:263). Agent names are case-sensitive and not a `Choice` (cli.py:121). `agent` hardcodes the task as "CLI agent run" (cli.py:165). | Tracebacks instead of usage errors, and remote data sources can't be used from the CLI at all. | Use `click.Choice(case_sensitive=False)`, write a ParamType that accepts URLs, add `-t` to `agent`, and add shell completion. |
| 6 | Every invocation has side effects and is slow to start. The group callback (cli.py:33-35) calls `get_settings()`, which creates `outputs/`, `temp/`, `logs/` and `checkpoints/` in the current directory (config.py:269-279), even for `agentds run --help`. `from agentds import __version__` (cli.py:16) imports the whole stack: litellm, langgraph, polars and sklearn (`__init__.py:30-36`). | Running `--help` litters the directory, and every command pays a heavy import cost. | Initialize lazily in the commands that need it, read the version with `importlib.metadata`, and create directories only when writing to them. |
| 7 | Log lines and command output are mixed on stdout. Logs are JSON by default (config.py:237) and go to `sys.stdout` (logger.py:160). No file handler is ever configured (cli.py:35 and app.py:438 pass no `log_file`), yet the docs point to `logs/agentds.log` (TROUBLESHOOTING.md:498, USER_MANUAL.md:512) and the web Logs tab (app.py:371-376) always shows "No logs available". | Breaks piping such as `agentds jobs \| ...`, and the documented log location doesn't exist. | Send logs to stderr, use text format on a TTY, and add a file handler at `settings.log_dir`. |
| 8 | The documented exception hierarchy is never raised. API_REFERENCE.md:542-572 shows catching `ValidationError` and `PipelineError` from `run()`, but agents raise `ValueError` and the gateway raises raw litellm errors or `RuntimeError` (llm_gateway.py:236). The API error body `{"error":"Job not found","detail":...}` (webhooks.py:100-105) does not match the documented `{"error":"JOB_NOT_FOUND","message":...}` (API_REFERENCE.md:513-538), and an unknown agent returns 400 where the docs say 404. `MissingAPIKeyError` builds wrong variable names (exceptions.py:76 produces `TOGETHER_API_KEY`). | Clients can't branch on stable error codes, and the documented error handling never triggers. | Raise the typed errors with a `hint` field, and use `to_dict()` for all API errors. |
| 9 | The Python library API has traps. The default `human_in_loop=True` makes `AgentDSPipeline().run()` a silent no-op. `run()` returns an untyped dict with any error buried in `result["state"]`. `Settings(debug=True)` (CONTRIBUTING.md:227, conftest.py:67-72) is probably ignored silently: those fields have aliases but `Settings` lacks `populate_by_name` (config.py:211-216), unlike `LLMSettings` (config.py:27). | Library users get no outputs and no error, and test fixtures don't apply the values they appear to set. | Return a typed `PipelineResult` with `.ok` and `.raise_for_error()`. Set `populate_by_name=True`. Default to no HITL when no approval channel exists. |
| 10 | The default `run` phases include deploy, there is no dry run, and Ctrl+C is not handled. The defaults are build plus deploy (cli.py:70), and CloudDeployAgent builds and starts Docker containers (cloud_deploy.py:160-198). Ctrl+C produces a raw traceback because only `Exception` is caught (cli.py:115). | A first `run --no-hitl` can start containers without being asked. | Default to `-p build`, add `--dry-run`/`--preflight-only`, and handle `KeyboardInterrupt` with exit code 130. |
| 11 | The shipped llm_config.yaml needs four providers. `agent_llm_mapping` routes agents to Groq, Google, Anthropic and OpenAI models (config/llm_config.yaml:32-100). Fallback triggers only on RateLimit, APIError or Timeout (llm_gateway.py:306), so a missing provider key does not fall back. | Users with a single key, which USER_MANUAL.md:31 says is enough, get some agents failing. | Default every agent to `default_model` unless the user overrides it, fall back on `AuthenticationError`, and validate model IDs in `status`. |
| 12 | The Docker setup drops keys and the CLI. Compose passes only three provider keys to `web` (docker-compose.yml:27-29), two to `api` and `worker`, and has no `env_file`. The image never installs the package, so the `agentds` CLI is absent (Dockerfile:50-54, 77). | Keys for other providers silently go missing inside containers, and the CLI can't be used there. | Add `env_file: ../.env` and run `pip install .` in the image. |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| 1 | Add `--json` to `status`, `jobs` and `run`, plus `--limit` on `jobs` | Makes output usable in scripts and CI |
| 2 | Show progress per agent during `run` (rich is already a dependency) | Pipelines take minutes and currently print nothing |
| 3 | Tell `agentds jobs` users when the queue is in-memory (job_queue.py:297-298) | Without Redis the list is always empty across processes |
| 4 | Fix the test docs: CONTRIBUTING.md:197 `-m unit` selects no tests, and :203 `-m "not slow"` replaces the default `addopts` marker filter, so LLM and integration tests run | Contributors hit confusing test runs |
| 5 | Make CONTRIBUTING.md:140 ("88 characters") match pyproject's `line-length = 100` | Contradictory style rules |
| 6 | Declare `click` as a direct dependency (it is imported at cli.py:14 but not listed in pyproject.toml) | It currently arrives only transitively |
| 7 | Quote APO_GUIDE.md:130 `pip install agentlightning[apo]>=0.2.0`, and drop the claim that Agent Lightning is used (apo.py:500-505 is a placeholder) | Unquoted, the shell treats `>` as a redirect; the claim is false |
| 8 | Default `web` and `api` to `127.0.0.1` and honor `HOST`/`PORT` from settings (cli.py:39-40, 52-53 vs config.py:225-226) | Safer local default and consistent precedence |
| 9 | Remove "ConfigManager" from the README diagram (README.md:170); no such class exists | Accuracy |

#### Error Message Audit
| Error Scenario | Current Message | What Developer Needs | Suggested Message |
|---------------|----------------|---------------------|-------------------|
| `run` with HITL on, or an agent failed | `[OK] Pipeline completed successfully!` | The truth and a non-zero exit code | `[ERROR] DataCleaningAgent failed: <reason> (job 3f2a…). Partial outputs: outputs/3f2a…/` · `[WARN] HITL not supported in CLI; use --no-hitl` |
| Preflight model unreachable | `Pre-flight checks failed: LLM connection failed` | Which model failed and how to change it | `Preflight: cannot reach 'openai/gpt-4o-mini' (missing OPENAI_API_KEY). Set LLM_DEFAULT_MODEL (e.g. ollama/llama3.1:8b) or add the key to .env` |
| Bad `-p` value | Traceback: `ValueError: 'Build' is not a valid PipelinePhase` | A usage error | `Invalid value for '-p': 'Build' is not one of 'build', 'deploy', 'learn'` |
| `agent AutoMLAgent file.parquet` | `No input from FeatureEngineerAgent` | What to run first, or which flag to pass | `AutoMLAgent needs FeatureEngineerAgent output: run 'agentds run … -p build' or pass a prepared parquet with --target COL` |
| `.env` copied verbatim | `SettingsError: error parsing value for field "api_keys" from source "DotEnvSettingsSource"` | The expected format | `API_KEYS must be comma-separated or a JSON list; leave empty to disable API auth in development` |
| No Logfire account | `You are not authenticated. Please run 'logfire auth'…` | To keep going | `[WARN] Logfire disabled (no LOGFIRE_TOKEN); using local logs only` |
| Poll status right after start | 404 `{"error":"Job not found","detail":"No job with ID: …"}` | The correct status (fix the bug); a stable error code | `{"error":"JOB_NOT_FOUND","message":"…","details":{"job_id":…}}` |
| `GET /api/jobs?status=bogus` | Unhandled `ValueError`, returned as 500 (webhooks.py:379) | A 422 listing valid values | `status must be one of pending, queued, running, paused, completed, failed, cancelled` |
| `status` with Ollama not running | `[OK] ollama` | Whether it is actually reachable | `[WARN] ollama: not reachable at http://localhost:11434 (run 'ollama serve')` |
| `run s3://bucket/x.csv` | `Path 's3://bucket/x.csv' does not exist.` | URLs accepted, or told where they are | Accept URLs, or `Remote sources: use the API/web UI` |
| AutoML can't find target | `Missing data_path or target_column` | How to specify the target | `Could not infer target column; pass --target or name it in the task (columns: a, b, c)` |
| Web UI HITL buttons | `No active pipeline` (always) | Working controls | Wire up `job_id_state`, then: `Approved DataCleaningAgent; continuing` |

#### Developer Journey
- **First use**: `pip install -e .` works, but USER_MANUAL's `-r requirements.txt` path leaves no `agentds` command. Copying `.env.example` likely breaks settings loading (`API_KEYS`). Even with a clean `.env`, any command, including `run --help`, can crash on Logfire auth. If a developer gets past both, the README Python snippet raises `TypeError` and the README CLI command prints `[OK]` without doing anything.
- **Daily use**: `--no-hitl` is effectively required but nothing says so. `-o` is ignored. Agent failures are reported as success. `agentds jobs` is empty without Redis. Logs are JSON interleaved with `[INFO]` lines on stdout. Editing `pipeline_config.yaml`, presets or `default_model` changes nothing. Single-agent runs fail for 8 of 10 agents.
- **Power use**: the REST "start, then poll" flow 404s, the SSE stream doesn't exist, HITL resume can't work because the graph was never invoked, and the typed exceptions are never raised. There is no `--json` output, no dry run and no shell completion. Contributors are blocked by the missing pre-commit config and the nonexistent `develop` branch.

#### Verdict Details
- Error Quality: CONCERNS
- Defaults & Config: CONCERNS
- Keyboard Experience: PASS (not applicable: no shortcut or editor surface; the CLI is keyboard-driven by nature; Gradio keyboard accessibility belongs to the a11y lens)
- Discoverability: CONCERNS
- Workflow Integration: CONCERNS

#### Confidence notes
- **C3 (`API_KEYS` parse failure)**: not executed, because pydantic-settings isn't installed. It rests on pydantic-settings parsing `list[str]` values from env and `.env` as JSON before validators run, a known behavior that `NoDecode` (2.7+) exists to work around.
- **C4 (Logfire)**: not executed. It rests on the Logfire SDK raising "not authenticated" when `send_to_logfire=True` and there is no token or credentials file. conftest.py:20 setting `ENVIRONMENT=test` supports this but doesn't prove it.
- **R3 (nested `.env`)**: depends on pydantic-settings and on LiteLLM's import-time `load_dotenv`. It may work by accident when the venv sits inside the repo.
- **R9 (`Settings(debug=True)` ignored)**: inferred from alias behavior without `populate_by_name`; not executed.
- **R11**: whether model IDs like `google/gemini-2.5-flash` and `groq/llama-4-scout` are valid in LiteLLM was not checked live.
- **R6**: startup cost was not measured. That click runs the group callback before a subcommand's `--help` is standard click behavior, but I did not run it here.
- **C6**: the 404 on status after start comes from reading the code (pipeline.py:526 calls `create_job` without the ID), not from running the server.

---

## 9. Performance Engineer

**Verdict**: NEEDS REWORK

*Performance Audit: Repo Health — main@d29bbd5*

**Perspective**: Performance Engineer
**Date**: 2026-11-03
**Repo Health — main@d29bbd5**: /home/user/AgentDS (General Repo Health mode; no diff or plan; working tree clean; nothing executed because project dependencies are not installed)

#### Repo Health Summary — main@d29bbd5
AgentDS runs long-lived servers (a Gradio UI and a single-process Litestar API) in front of a CPU- and LLM-heavy LangGraph pipeline. The main problems are:
- Synchronous pipeline, LLM and Redis work runs inside `async` handlers.
- The health probe makes a paid LLM call.
- The pipeline registries are never evicted.
- A shared-list mutation in the LLM gateway turns provider outages into retry storms.
- On the data side, a dense one-hot encode has no cardinality guard, and DataFrames are copied repeatedly through pandas, disk and graph state.

#### Files Reviewed
| File | Performance Role | Risk |
|------|-----------------|------|
| agentds/core/llm_gateway.py | Every LLM call: retry, fallback, cost | High |
| agentds/web/api/webhooks.py | Litestar handlers, health-probe target, pipeline registry | High |
| agentds/web/app.py | Gradio callbacks, pipeline registry | High |
| agentds/workflows/pipeline.py | Per-pipeline object graph, compile, state, MemorySaver | High |
| agentds/core/job_queue.py | Redis job registry (pickle hash) | High |
| agentds/agents/feature_engineer.py | One-hot encoding, pandas round-trip | High |
| docker/docker-compose.yml, docker/k8s/deployment.yaml, docker/Dockerfile | Probes, process model, Redis eviction policy | High |
| agentds/core/artifact_store.py | Per-artifact copy, checksum, metadata rewrite | Medium |
| agentds/agents/data_cleaning.py, eda_copilot.py | Full-frame scans; DataFrame put in outputs | Medium |
| agentds/agents/automl.py | Optuna × CV × XGBoost CPU | Medium |
| agentds/agents/data_loader.py | Ingest from file, cloud, DB, API | Medium |
| agentds/core/config.py, cache_layer.py | Redis client setup, YAML reloads | Medium |
| agentds/agents/api_wrapper.py | Generated inference API template | Medium |
| agentds/web/api/middleware.py | Auth plus token bucket per request | Low (bounded) |
| agentds/agents/base.py, optimization.py, drift_monitor.py, cloud_deploy.py, devops.py; agentds/core/apo.py | LLM call sites, sleep-poll | Low |
| agentds/integrations/notifications.py, n8n.py | Outbound HTTP and SMTP (not wired in) | Low |
| agentds/core/logger.py, agentds/cli.py, package `__init__.py` files | Logging, startup imports | Low |

#### Verdict: NEEDS REWORK

#### Critical Issues (Must Fix)
| # | Issue | File(s) | Impact | Fix |
|---|-------|---------|--------|-----|
| 1 | **Fallback chain grows on every LLM call.** `get_fallback_chain` returns the list from the YAML itself (`models = self.fallback_chains.get(chain_name, [])` → `FallbackChain(models=models)`). `complete()` then calls `fallback.models.insert(0, model)` on that list, so each call adds one entry to the shared `default` chain. The config ships in the image (Dockerfile:52), so this happens in production. | llm_gateway.py:178-181, 229-230, 233-236, 305-313, 183-187 | One gateway is shared by all agents and kept across HITL resumes (pipeline.py:142, 160-164). During an outage, the loop first retries every model this gateway has ever used, then reaches the configured fallbacks. Each `litellm.Timeout` costs up to `request_timeout`=120s, so ~14 entries after one build+deploy run means ~28 min stuck in one call. On `RateLimitError`, tenacity repeats the whole walk 3×, sending ~40+ requests to a provider that is already rate-limiting. The chain also keeps growing. | Copy the list (`FallbackChain(models=list(models))`), dedupe the primary, and cap total attempts. Keep one retry layer (either tenacity or the fallback walk, not both) and use jittered backoff. |
| 2 | **Blocking work inside `async def` handlers on a single-process server.** `health_check` makes a sync `validate_connection()` call (litellm `completion` plus tenacity `time.sleep`). `start_pipeline` builds and runs the pipeline; with `human_in_loop=False` that is the full `compiled.invoke`. `pipeline_action` calls `pipeline.resume()`, which also does `compiled.invoke`. `run_agent` calls `agent.run()` (polars, sklearn, Optuna, Docker build, and a `time.sleep` poll loop). Every job endpoint uses sync Redis. Everything is served by `uvicorn.run(app, ...)` in one process. | webhooks.py:142, 171-179, 285, 348, 211/239/380/403/421, 521; pipeline.py:560-564, 621; docker-compose.yml:56; k8s deployment.yaml:234 | One pipeline or agent request freezes the event loop for minutes to hours. Every other request queues behind it, including `/api/health`. The k8s liveness probe (default 1s timeout) then fails, the pod restarts, and the in-flight run is lost. | Make short handlers plain `def` (Litestar `sync_to_thread=True`) or use `redis.asyncio`. Send pipeline and agent runs to the RQ worker that compose already deploys (docker-compose.yml:164) and return 202 plus `job_id`. |
| 3 | **Health endpoint makes a paid LLM call on every probe.** Each request runs `gateway = LLMGateway(settings)` (YAML parse plus `os.environ` writes) and then `gateway.validate_connection()`, which is a real completion. | webhooks.py:137-147; k8s deployment.yaml:259-270 (readiness 10s, liveness 30s, no `timeoutSeconds`); docker-compose.yml:73-74 | About 8 LLM calls per minute per replica (at least 960/h at `minReplicas: 2`), burning rate-limit quota. Any LLM latency over 1s fails the probe, so the pod is pulled from service or restarted even when the API is healthy. A provider outage makes the API look dead. | Split into `/live` (process only) and `/ready` (Redis PING with a timeout). Run LLM reachability as a cached background check (for example every 5 min), never inline in a probe. |
| 4 | **`_active_pipelines` registries are never evicted.** Each entry pins a whole `AgentDSPipeline`: LLMGateway, an ArtifactStore whose `_artifacts` is an in-memory copy of all historical artifact metadata, a JobQueue with its own Redis pool and connection (`redis.from_url` plus PING), 10 agents, and a MemorySaver holding every checkpoint. The API entry also pins the final `result` state. | app.py:26, 290-294; webhooks.py:112, 182-185; pipeline.py:142-152; job_queue.py:203, 289; artifact_store.py:81 | Memory grows with every Start click or `POST /pipeline/start`. Growth becomes super-linear once artifacts build up, because each new store loads all prior metadata. Each start also holds one Redis connection that is never released. Because HITL is on by default (app.py:111-114, webhooks.py:42), every start leaks even when nothing ran. End result: pod OOM (API limit 2Gi, web 4Gi) or Redis `maxclients` over days. | Bound the registry (TTL/LRU), delete entries on completion or stop, and keep resumable state in a persistent checkpointer keyed by `job_id` instead of pinning the object. Share one gateway, store and queue per process. |
| 5 | **Dense one-hot encoding with no cardinality guard.** The fallback plan takes every string column (`categorical_cols = [c for c in df.columns if df[c].dtype == pl.Utf8]`, first 10); the LLM's plan is not checked either. Encoding uses `OneHotEncoder(..., sparse_output=False)`, after `df.to_pandas()` and a `pdf.drop` copy, and the result is copied again into `pl.DataFrame(X, ...)`. | feature_engineer.py:227-228, 304-308, 342, 347, 353, 366 | Memory needed is rows × total categories × 8 B. A 200k-row CSV with a string `customer_id` or email column needs ~320 GB, so the worker is OOM-killed. | Filter categoricals by `n_unique` (absolute or percentage cap), drop ID-like columns, and use `sparse_output=True` with `max_categories`/`min_frequency`. Keep the matrix sparse into XGBoost. |

#### Recommendations
| # | Issue | File(s) | Why | Fix |
|---|-------|---------|-----|-----|
| 1 | Polars DataFrame placed in agent outputs: `"dataframe": df_cleaned` | data_cleaning.py:168 → pipeline.py:331 (`result.model_dump()` into graph state), 152, 574-575 → job_queue.py:234 | The whole dataset flows into LangGraph state and checkpoints, `final_outputs`, the pickled Redis job record and `_active_pipelines[...]["result"]`. It is either copied many times or breaks checkpoint serialization. | Remove it. Downstream agents already re-read `data_path`. |
| 2 | Redis job registry costs O(history) per call | job_queue.py:212-216, 233-234, 251-256; callers app.py:357, webhooks.py:380, cli.py:273; docker-compose.yml:91 | All jobs live in one hash with whole `Job` objects pickled (including `result`). `hvals` plus unpickling every job runs just to show 50 rows. There is no TTL, and `update_job` takes 2 round trips. Under `allkeys-lru` the single key can be evicted all at once. | One key per job with EXPIRE, a ZSET index on `created_at`, then `ZREVRANGE 0 49` plus pipelined `MGET`. Store a result summary only, as JSON. |
| 3 | Redis clients ignore the configured timeouts and pool limits | job_queue.py:203; cache_layer.py:131; config.py:108-111 | `redis.from_url(url)` sets no `socket_timeout`, `socket_connect_timeout` or `max_connections`, although `RedisSettings` defines them. A Redis stall hangs callers forever (freezing the event loop in the API). Every JobQueue gets a new pool (pipeline.py:144, app.py:40, webhooks.py:121). | One shared `ConnectionPool.from_url(..., max_connections, socket_timeout, socket_connect_timeout, health_check_interval=30)`. |
| 4 | Heavy objects rebuilt per request | webhooks.py:141, 171, 331, 338-339; base.py:149-150; pipeline.py:557, 604; llm_gateway.py:99-140 | `run_agent` builds 2 gateways and 2 ArtifactStores: the agent creates its own because none is passed, so the context's pair is never used. Each gateway re-reads YAML and rewrites `os.environ`. Each store parses all of `.artifacts.json`. Each JobQueue reads the flags YAML and PINGs Redis. The graph is recompiled on every run and resume. | Build singletons at startup (Litestar `on_startup`/DI). Pass shared instances into agents. Compile the graph once in `__init__`. |
| 5 | ArtifactStore write amplification | artifact_store.py:96-103, 180, 197, 204, 240 | Every save or delete rewrites the whole metadata JSON with `indent=2`: O(N) per save, O(N²) over the store's lifetime. `shutil.copy2` duplicates each temp parquet or model file, then `_calculate_checksum` re-reads the copy in full. Several instances race on the same file. | Use a per-job metadata file or SQLite. Hash while copying, or `os.replace`/hard-link the temp file instead of copying. |
| 6 | No LLM response caching; preflight is a paid call | cache_layer.py:329-332 (no call sites); llm_gateway.py:82; pipeline.py:249 | The docstring promises caching, but nothing uses `CacheLayer`. Reruns resend identical temperature-0 prompts. Every run pays one preflight call, and it checks `default_model`, not the agents' models. | Cache completions by sha256(model, messages, temperature, max_tokens) when temperature is 0 (Redis or `litellm.cache`). Make preflight a key check or a cached ping. |
| 7 | Gradio progress is fake and concurrency is serialized | app.py:298-302, 308-319, 61-65 | The first `yield` comes only after `pipeline.run` finishes. Progress is then replayed with `time.sleep(0.5)` per agent (~5s added). No `concurrency_limit` is set, so Gradio's default of 1 makes all users wait behind one pipeline. | Enqueue to the worker. Stream real status (poll `get_job` with `gr.Timer` every 2-5s). Remove the sleeps and set `concurrency_limit`. |
| 8 | Eager, duplicated loading and disk build-up | data_loader.py:286-299, 308, 106-107; data_cleaning.py:138-139; feature_engineer.py:140-145; automl.py:146-148 | Cloud loads do `content = f.read()` then `io.BytesIO(content)`, doubling peak memory. The DB path goes `pl.from_pandas(cx.read_sql(...))`. Each stage writes a fresh parquet to `temp_dir` that is never deleted, and Gradio uploads are never purged. | Use `pl.scan_*/read_*` with `storage_options`; `cx.read_sql(..., return_type="polars")`; clean `temp_dir` in finalize; `gr.Blocks(delete_cache=(...))`. |
| 9 | Repeated polars ↔ pandas ↔ numpy conversions | feature_engineer.py:342-366; automl.py:126, 202-208 | Each conversion is a full copy. `.values` on mixed-dtype frames produces object arrays (slow XGBoost input). | `df.drop(target).to_numpy()`; `pipeline.set_output(transform="polars")` (sklearn>=1.4 is already pinned); pass arrays straight to AutoML. |
| 10 | AutoML compute is unbounded | automl.py:296-300, 321, 332-333 | `n_trials`/`cv_folds` come straight from LLM JSON with no cap. `study.optimize` has no `timeout` and no pruner, and runs full-data CV every trial. XGBoost's default thread count follows host cores, not the k8s CPU limit (1-2 cores), so it gets throttled. | Clamp trials and folds, add `timeout=` and `MedianPruner`, subsample during search, use `tree_method="hist"`, set `n_jobs` explicitly to the CPU limit. |
| 11 | Redundant full-frame scans in Cleaning and EDA | data_cleaning.py:111, 121, 193, 241, 320, 222/232/268; eda_copilot.py:179-185, 212-215 | `df.unique()` builds a deduped copy 4× per run. EDA computes each stat twice (`float(col.mean()) if col.mean() is not None`), which is 14 scans per numeric column, plus up to 190 separate `pl.corr` selects. | Use `df.is_duplicated().sum()`; one `df.select([...aggregates])` or `describe()`; `df.select(numeric_cols).corr()` once. |
| 12 | Generated inference API loops per row in batch predict | api_wrapper.py:380-405 (template) | For each instance it builds a 1-row DataFrame and calls `transform`, `predict` and `predict_proba`, inside `async def`: 3N model calls on the event loop. | Build one DataFrame for the batch with one transform and predict; run it in a thread. |

#### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| 1 | Make package exports lazy (agentds/__init__.py:29-35, agents/__init__.py, core/__init__.py) | `cli.py:16` imports `agentds` just for `__version__`, which pulls in litellm, langgraph, sklearn, polars and all agents. `agentds --help`, `status` and `jobs` pay several seconds of imports (estimated). |
| 2 | Drop never-imported heavy dependencies: duckdb, llama-index, pydantic-ai, mlflow (pyproject.toml:23-25, 43, 54-58) | No `import` of any of them in agentds/ or tests/. They make the image larger and slower to pull. |
| 3 | Cache feature flags (config.py:296-311) | `is_feature_enabled` re-parses YAML on every call. It is called rarely today, but it is a trap. |
| 4 | Read only the log tail (app.py:373-375) and rotate logs (logger.py:173) | `f.readlines()[-100:]` loads the whole file. Use `deque(f, maxlen=100)` or seek from the end; use `RotatingFileHandler`. |
| 5 | `EmailNotifier.send` is `async` but uses blocking `smtplib` with no timeout (notifications.py:225-234); `CompositeNotifier` sends one by one (259-269) | Not wired in yet. Use `aiosmtplib` or `to_thread` with a timeout, and `asyncio.gather`. |
| 6 | Run LLM calls in parallel in OptimizationAgent (optimization.py:153-157) and APOOptimizer (apo.py:331-349); reset `self.beam` per `optimize()` (apo.py:288) | Up to 18 and ~45 serial calls. `acomplete` plus `gather` cuts wall time. Stale beam candidates carry over between runs. |
| 7 | Bring `acomplete`/`astream` in line with `complete` (llm_gateway.py:360-367, 437-443) | `astream` has no timeout and neither has fallback; both are unused today. |
| 8 | Log "No API keys configured" once (middleware.py:217-220) | Today it logs a warning on every request in dev mode. |
| 9 | Remove duplicate report writes (drift_monitor.py:128-138, optimization.py:163-172) | The same JSON is written to `temp_dir` and again through `save_artifact`. |
| 10 | Use either the idle RQ worker or remove it (docker-compose.yml:164; `RedisQueue._queue` at job_queue.py:204 is never used) | It runs a process that does nothing. It is the natural home for the fix to Critical #2. |

#### Resource Budget
| Resource | Current | After Recommended Fixes | Concern? |
|----------|---------|-----------|----------|
| LLM calls per action (IPC analogue) | ~9 per build+deploy run (1 preflight + 8 agents), up to +18 with learn phase; +1 per `/api/health` probe (~8/min per k8s replica); in an outage, (5 + calls served so far) attempts per call, ×3 on rate limits | ~8 per run, cache hits on reruns, 0 on probes, at most ~6 attempts per call | Yes |
| Redis round trips per action | create = 1 HSET; update = 2 (HEXISTS+HSET); list = HVALS of the whole history plus unpickling all of it; new pool + PING per pipeline | list = ZREVRANGE + MGET(50); update = 1; one shared pool | Yes |
| Connections, pools, background tasks (listener analogue) | +1 Redis pool and connection per pipeline started, never released; RQ worker idle; no background tasks | 1 bounded pool per process; worker runs the pipelines | Yes |
| Memory growth per hour | Unbounded: one full pipeline object (gateway, artifact-metadata copy, MemorySaver, Redis connection) per start, plus dataset copies in state and job records | Flat (TTL-bounded registry, persistent checkpointer, no DataFrames in state) | Yes |
| Startup time impact | `import agentds` eagerly loads litellm, langgraph, sklearn, polars and all agents; `Settings()` runs mkdirs | Lazy exports | Minor |
| Bundle size impact (image and dependency weight) | One image for web, API and worker, including unused duckdb, llama-index, pydantic-ai, mlflow | Trimmed dependencies | Minor |

#### Hot Path Analysis
- **Health probes (most frequent path):** every probe runs a gateway constructor, a YAML parse, `os.environ` writes and a real LLM completion with sync tenacity sleeps, all on the event loop (Critical #2 and #3). This alone can take down the API.
- **LLM call path** (`BaseAgent.call_llm` → `LLMGateway.complete`): synchronous, no cache. It mutates the shared fallback list (Critical #1), then opens a logfire span, writes 2 structlog lines and runs `completion_cost`. That overhead is negligible next to network latency; the retry structure is the real risk.
- **API request path:** the middleware (auth set lookup plus token bucket) is O(1) and its state is bounded. The handlers behind it do blocking work (Critical #2).
- **Per-stage data path:** read parquet → compute → write a new parquet to `temp_dir` → `copy2` into outputs → MD5 re-read → rewrite all metadata JSON. That is about 3-4× the dataset's I/O per stage, and Cleaning and EDA add several more full scans (Recommendation #11).
- **AutoML:** the main CPU cost is `n_trials × cv_folds` XGBoost fits with no cap or timeout (Recommendation #10). In the API it runs on the event loop.
- **Gradio:** Start runs the pipeline synchronously and then fakes progress. Refresh Jobs unpickles the entire job history.
- **Checklist items with no analogue or no finding:**
  - DuckDB is listed in the stack but is never imported, so there is nothing to audit.
  - No file watchers and no SSE: the `sse_updates` flag is unused and `astream` has no callers.
  - No unbounded busy-wait. The only sleep loops are `cloud_deploy.py:203-214`, capped at about 30 iterations (~90s worst case), and the Gradio fake progress.
  - The rate-limiter state is bounded: idle buckets are cleaned every 5 min (middleware.py:112-130). It is per process, though, so the effective limit multiplies with replica count.
  - `MemoryCache` is capped at 1000 entries (cache_layer.py:86-91) and is not used.

#### Verdict Details
- Memory Management: CONCERNS (Critical #4 and #5, Recommendations #1 and #2)
- IPC Efficiency (LLM, Redis, HTTP): CONCERNS (Critical #1 and #3, Recommendations #2 and #6)
- I/O Patterns: CONCERNS (Recommendations #5 and #8)
- Startup Impact: CONCERNS, minor (Nice-to-haves #1 and #2)
- Long-Session Stability: CONCERNS (Critical #2 and #4, Recommendation #3)

#### Confidence notes
- **Critical #1:** the list mutation is certain from the code. The timing figures assume `litellm.Timeout` is not a subclass of `litellm.APIError`, so tenacity does not retry timeouts while the inner loop does, and that `RateLimitError` is retried by both layers. That matches litellm's exception hierarchy as I know it, but I did not run it.
- **Critical #3:** assumes k8s default probe settings (`timeoutSeconds: 1`, `failureThreshold: 3`), since none are set for the API. The call counts are arithmetic from the configured periods.
- **Critical #4:** "super-linear" growth only applies once artifacts build up in the shared `.artifacts.json` (non-HITL runs, CLI, `/agent/run`). The linear leak, including one Redis connection per start, applies to every start.
- **Critical #5:** the memory figure is an estimate and depends on the column chosen. The fallback plan definitely includes every string column; whether the LLM plan does is model-dependent.
- **Recommendation #1:** what LangGraph's checkpoint serializer does with a polars DataFrame depends on the version and `pickle_fallback`. It either raises (breaking non-HITL runs with checkpointing) or stores full copies. Either way the DataFrame should not be in the outputs.
- **Recommendation #7:** Gradio's default concurrency of 1 per event comes from Gradio 4/5 behaviour, not from running the app.
- **Nice-to-have #1:** the import-time cost is estimated; I could not measure it without the dependencies installed.

---

## 10. Accessibility Engineer

**Verdict**: APPROVE WITH CHANGES

**Perspective**: Accessibility Engineer · **Date**: 2026-11-03 · **Source**: Repo Health — main@d29bbd5

### Summary
Every interactive element is a stock Gradio component (native inputs, buttons, tabs, file upload with labels), which gives a reasonable keyboard and screen-reader baseline. The gaps come from how the app uses those components: a slider standing in for progress, status updates that aren't announced, skipped heading levels, and colour-only status styles that are defined but not yet applied.

### Files Reviewed
| File | A11y Role | Risk |
|------|----------|------|
| `agentds/web/app.py` | All interactive UI | Medium |

### Critical Issues (Must Fix)
None. The UI's blocking problems (hidden or unwired controls) are behavioural and are covered under UX/Frontend. Nothing here locks a keyboard or screen-reader user out of a flow that works for mouse users.

### Recommendations
| # | Issue | WCAG Criterion | Fix |
|---|-------|---------------|-----|
| A1 | The progress bar is a non-interactive `gr.Slider` (`app.py:136-142`). Assistive tech announces it as a (disabled) slider, not as progress | 4.1.2 Name, Role, Value | Use `gr.Progress()` or an HTML `<progress aria-label="Pipeline progress">` |
| A2 | Status, agent output and error text update in `gr.Markdown` (`app.py:133, 145`) with no live-region semantics, so screen-reader users aren't told when a run fails or completes | 4.1.3 Status Messages | Send key transitions through `gr.Info`/`gr.Warning` toasts, or render status in `gr.HTML` with a `role="status"` wrapper |
| A3 | The heading hierarchy skips levels: `# AgentDS` (h1) → `### Autonomous…` (h3) → `### Configuration`, then `#### Build Phase` (h4) with no h2 anywhere (`app.py:72-76, 85, 130, 184`) | 1.3.1 Info and Relationships | Use `##` for the tab-section headings and `###` for the sub-groups; make the subtitle plain text |
| A4 | The (currently unused) status classes communicate state by colour alone, and on a white background they fail contrast: green `#22c55e` ≈ 2.3:1, amber `#f59e0b` ≈ 2.2:1, blue `#3b82f6` ≈ 3.7:1, red `#ef4444` ≈ 3.8:1 (`app.py:44-47`) | 1.4.1 Use of Color; 1.4.3 Contrast (Minimum) | Before applying them, pair each with a text label or icon and use theme colours with ≥4.5:1 contrast |
| A5 | `gr.Markdown(label="Current Agent")` (`app.py:145`): Markdown labels aren't rendered, so the region has no accessible name | 2.4.6 Headings and Labels | Put a visible `####` heading inside, or use `gr.Textbox(label=…, interactive=False)` |

### Nice-to-Haves
| # | Suggestion | Rationale |
|---|------------|-----------|
| A6 | When review controls become visible (after the UX fix), move focus to "Approve & Continue" or announce "Review required for {agent}" | Dynamic UI changes need focus management or an announcement (2.4.3) |
| A7 | Disable and re-label Start during a run ("Running…") | Clear state for assistive tech and keyboard users |

### Keyboard Navigation Walkthrough
| Step | Action | Expected Focus | Expected Announcement | Handles? |
|------|--------|---------------|----------------------|----------|
| 1 | Tab into the page | First tab ("Pipeline") | "Pipeline, tab, selected" | Yes (Gradio tabs) |
| 2 | Tab | File upload | "Upload Data File, button" | Yes |
| 3 | Tab ×2 | URL, then Task textboxes | Labels read | Yes (`label=` set) |
| 4 | Enter on "Advanced Options" | Accordion toggles | Expanded/collapsed | Yes (Gradio accordion) |
| 5 | Tab to Start, press Enter | Start button | "Start Pipeline, button" | Yes |
| 6 | Run completes or fails | Focus stays on Start | Status change should be announced | **No** (A2) |
| 7 | Review needed | Review buttons | "Review required" | **No** (controls never shown; A6) |
| 8 | Tab to the progress "slider" | Slider | "Progress, slider, 0, disabled" | Misleading (A1) |

### ARIA Checklist
| Element | Role | Label | States | Keyboard | Notes |
|---------|------|-------|--------|----------|-------|
| Tabs | tab/tablist (Gradio) | Visible text | selected | Yes | OK |
| File upload | button (Gradio) | "Upload Data File" | — | Yes | OK |
| Textboxes | textbox | Visible labels | — | Yes | OK |
| Start / Refresh buttons | button | Visible text | No busy or disabled state during a run | Yes | A7 |
| Progress | slider (wrong role) | "Progress" | disabled | Focusable | A1 |
| Status / Current Agent | none | none | not live | — | A2, A5 |
| Review buttons | button | Visible text | Hidden permanently | — | UX U3 |

### Verdict Details
- Screen Reader Compatibility: CONCERNS (no status announcements; wrong progress role)
- Keyboard Navigation: PASS (stock Gradio components)
- Focus Management: CONCERNS (no handling for dynamic review state)
- Color Contrast: CONCERNS (latent colour-only, low-contrast status classes)
- Motion Sensitivity: PASS (no custom animation)
