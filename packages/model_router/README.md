# Model Router Proxy

OpenAI-compatible routing proxy for explicit persona/runtime model routing.

Purpose:

- A host agent points to this proxy as its model API base URL.
- The host uses a virtual model name such as `persona-auto`.
- The proxy chooses a real upstream model per request from explicit metadata.
- Route logs must never contain raw prompts, API keys, or headers.

Files:

- `app.py` — proxy server.
- `config.example.yaml` — routing config example.
- `tests/test_router.py` — local tests.

This public edition ships example configuration only. Deployment credentials and route logs are intentionally excluded.
