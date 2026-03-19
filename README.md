# Customer Support Voice Agent

A customer support agent built with the OpenAI Agents SDK, deployed on Databricks Apps.

## Overview

This agent helps customers with:
- Order status and tracking
- Returns and refunds
- Product information
- Account management
- Billing and payment questions
- Technical support

## Setup

```bash
uv run quickstart --profile dogfood
```

## Run Locally

Start the full app (text chat UI + voice UI):

```bash
uv run start-app
```

Or start just the backend server (sufficient for the voice UI):

```bash
uv run start-server
```

The server starts on port 8000 by default. To use a different port:

```bash
uv run start-server --port 8100
```

### Voice UI

Once the server is running, open **`http://localhost:8000/voice`** in your browser.

- Click **Connect** — the browser will ask for microphone permission, then open a WebSocket to the server.
- Speak naturally. The agent listens via server-side VAD and responds with synthesized speech (voice: *Alloy*).
- The agent's transcript appears in real time below the controls.
- Click **Disconnect** to end the session.

The voice relay requires an `OPENAI_API_KEY` with access to `gpt-4o-realtime-preview`. Add it to your `.env` file:

```
OPENAI_API_KEY=sk-...
```

### Run Tests

```bash
uv run pytest tests/ -v
```

This runs:
- **`tests/test_voice_relay.py`** — unit tests with a mocked OpenAI connection (tool logic, relay behaviour, session config).
- **`tests/test_voice_integration.py`** — live tests against a real server instance (HTML content, WebSocket acceptance, JSON event format).

## Deploy

```bash
databricks bundle deploy
databricks bundle run agent_customer_support
```

After deploying, visit `<app-url>/voice` for the voice UI. Set `OPENAI_API_KEY` as a secret in your Databricks App configuration so the relay can reach the OpenAI Realtime API.
