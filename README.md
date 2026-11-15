# Free Model Router

[![make.fun](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/bactiar291/free-model-router-makefun/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard)
[![reward](https://make.fun/badge/free-model-aggregator-01/bactiar291/free-model-router-makefun/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

An original deterministic Python entry for the [make.fun](https://make.fun) free-model-aggregator-01 competition. It discovers the ten simulated provider catalogs, chooses eligible free models using the supplied quality ratings and remaining workload, and exposes OpenAI Chat Completions and Anthropic Messages in JSON and SSE.

Python standard library only. Targets Python 3.13; tested locally with Python 3.12. No packages, external API keys, real model APIs, or make.fun network calls are needed by the entry.

## Run

The harness supplies MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL. With a running local competition env:

```sh
MAKEFUN_PROMPT_FILE=/path/to/practice/input/prompt.txt \
MAKEFUN_ENV_URL=http://127.0.0.1:ENV_PORT \
python3 -P -s main.py
```

The server binds 0.0.0.0 on an OS-assigned port, registers protocol version 1 and its catalog, serves the workload, polls /work, and exits successfully when done. The caller must supply the actual env port.

```sh
python3 main.py --list
```

Outside a trial this prints exactly `{"providers":[]}` followed by a newline. With both trial variables set, it probes the catalogs and prints the same compact catalog object used in registration, without starting a server. Keys are never included in that object.

Endpoints:

- POST /v1/chat/completions: OpenAI JSON or SSE.
- POST /v1/messages: Anthropic JSON or SSE.
- GET /v1/models: Anthropic list for x-api-key; OpenAI list otherwise. Duplicate IDs across providers appear once in the free-model union; the registered catalog retains every provider and every free or paid model.

## Routing and wire behavior

Catalogs establish availability and free flags; the prompt supplies rating_milli and capabilities. Authentication follows each provider's declared scheme independently of its wire. Provider URLs must remain on the supplied env origin under /p/. HTTP redirects and proxy environment variables are not used.

The router filters capabilities and JSON-wire compatibility, then plans the remaining workload over provider call slots with a deterministic assignment solver. Request weights and model ratings determine the quality cost; capability constraints reserve vision and long-context options. A quality-loss allowance of 1.8% targets at least 98.2% weighted quality while reducing expected refusals. The published scorer gives total 99 for quality 98 or 99 when other components are perfect, so choosing slightly lower-rated real models can improve call-count tie breaks.

Risk estimates use the public generator's per-provider budget ranges and failure/outage distributions. These are priors, never actual instance budgets or schedules. First calls have no scheduled failures; observed temporary refusals lower the estimated number of remaining scheduled failures. The planner recomputes after responses, uses stable identity/column ties, and makes provider calls sequentially. Quality-first ranking remains the fallback for briefs without a complete quality budget.

A refusal fails over to another eligible provider. Explicit exhausted-budget/outage messages or permanent auth errors retire a provider; temporary rate, payment, queue, and concurrency refusals do not. If all alternatives fail, one bounded recovery pass can retry explicitly temporary refusals. A successful inference is never replaced with a second inference. Malformed successful responses produce an error instead of fabricated content.

Wire translation preserves upstream text, tool arguments and IDs, token usage, stop reasons, and request system/text/image/tool/result/JSON/stop/output-limit fields supported by the competition. OpenAI streams request final usage. Anthropic streams emit message_start, ordered content blocks, message_delta, then message_stop. Gemini streams use streamGenerateContent. All content comes from actual provider responses.

Runtime reads only its own source and the designated prompt; it never reads instance seeds, provider budgets, workload bodies, state exports, or reference files. The separate test parent supplies those files only to the public env/scorer/generator programs.

## Tests

Run from the repository root. Integration tests require Node (the kit targets Node 22+) and the unchanged public kit; no installation is required.

```sh
python3 -m unittest discover -s tests -v
python3 tests/integration.py --kit /tmp/makefun/free-model-aggregator-01-kit --repeat 2 --keep
python3 tests/integration.py --kit /tmp/makefun/free-model-aggregator-01-kit --repeat 2 --extra 8 --keep
```

The runner invokes program/env/server.mjs, program/scorer/score.mjs, and optionally program/generator/draw.mjs directly. It does not execute cli/makefun.mjs. Every env binds port 0, uses explicit temporary data/state/output paths below /tmp/makefun, and is terminated through its own subprocess handle to trigger the canonical export. Nothing uses global /input or /output paths. Omit --keep to delete successful test artifacts; failures retain their artifacts.

Each repeat uses a fresh env and port. The runner checks exit status, workload completion, --list stability before/after work, catalog equality with registration, official scorer output, and exact state.json byte equality. Extra practice seeds are SHA-256 hashes of public test labels, generated solely in the test parent and never passed to the router.

Measured after the call-efficient routing revision:

| Practice | Score | Quality | Served | Wire | Failover | Catalog | Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| trial-0 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |
| trial-1 | 99 | 98 | 100 | 100 | 100 | 100 | 23 |
| trial-2 | 99 | 98 | 100 | 100 | 100 | 100 | 22 |

HEAD scored 99 with 26/29/29 calls on the paired official practice runs. On 32 new development seeds, the selected policy averaged 22.28125 calls versus HEAD's 26.625, with score 99 and all requests valid. On 64 separate holdout seeds, it averaged 22.1875 calls versus 26.78125 for HEAD; every run scored 99 with served/wire/failover/catalog 100. True weighted quality averaged 98.507275% versus HEAD's 99.780871%, deliberately using the score floor to improve tie-breaking.

The selected policy produced 20 calls on some holdout seeds and 22–23 on all official practice seeds. Both versions ran twice per holdout and official instance with exact repeated state bytes and successful exits. The 1.9% cooldown policy without recovery was rejected because it caused a real 503 and score regression on development. These are local public-scorer results, not an official submission or transcript-gate verdict. The quality reserve cannot guarantee every hidden draw; 20-call ties still favor earlier receipts, and no placement is promised.

## Limits

SSE is buffered and validated before conversion: this adds first-token latency, but lets Anthropic message_start report accurate input usage even when OpenAI sends usage last. Request bodies are bounded to 8 MiB; upstream responses to 2 MiB; prompt input to 1 MiB. Network operations have finite timeouts. Streams missing their required finish events are rejected.

This is a competition router, not a general production API gateway. It supports one completion per request and the competition's content types; unsupported modalities such as audio and thinking blocks are rejected. It does not implement catalog pagination, public-client authentication, or production deployment controls. The simulated workload does not exercise every real-provider extension. Hidden final instances and the platform's eligibility/transcript checks remain unverified locally.

MIT licensed. This repository is the entry source; submission status is shown by the make.fun badges above.
