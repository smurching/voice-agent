"""
Tests for agent_server/voice_relay.py

Covers:
- Tool execution logic (_execute_tool)
- GET /voice returns the HTML page
- WebSocket /ws/voice relay: audio forwarding, tool call interception
"""

import asyncio
import base64
import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_server.voice_relay import _execute_tool, router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def silent_pcm16_b64(num_samples: int = 1024) -> str:
    """Return base64-encoded silent (all-zero) PCM16 audio."""
    samples = struct.pack(f"<{num_samples}h", *([0] * num_samples))
    return base64.b64encode(samples).decode()


# ---------------------------------------------------------------------------
# Unit tests: _execute_tool
# ---------------------------------------------------------------------------

class TestExecuteTool:
    def test_get_current_date_and_time_returns_string(self):
        result = _execute_tool("get_current_date_and_time", {})
        assert isinstance(result, str)
        assert "UTC" in result

    def test_check_return_eligibility_within_window(self):
        result = _execute_tool("check_return_eligibility", {"days_since_purchase": 5})
        assert "Eligible" in result
        assert "25 day(s)" in result

    def test_check_return_eligibility_exactly_30_days(self):
        result = _execute_tool("check_return_eligibility", {"days_since_purchase": 30})
        assert "Eligible" in result
        assert "0 day(s)" in result

    def test_check_return_eligibility_outside_window(self):
        result = _execute_tool("check_return_eligibility", {"days_since_purchase": 31})
        assert "Not eligible" in result
        assert "31 days since purchase" in result

    def test_unknown_tool(self):
        result = _execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result


# ---------------------------------------------------------------------------
# HTTP test: GET /voice
# ---------------------------------------------------------------------------

class TestVoiceUI:
    def test_get_voice_returns_html(self):
        app = make_app()
        with TestClient(app) as client:
            response = client.get("/voice")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "WebSocket" in response.text or "ws/voice" in response.text


# ---------------------------------------------------------------------------
# WebSocket tests (mocked OpenAI Realtime connection)
# ---------------------------------------------------------------------------

# Helper: build minimal Pydantic-like event objects that have .type and .model_dump()
def make_event(event_type: str, **kwargs):
    obj = MagicMock()
    obj.type = event_type
    obj.model_dump.return_value = {"type": event_type, **kwargs}
    return obj


def make_output_item_done_event(item_type: str, call_id: str = "cid1", name: str = "", arguments: str = "{}"):
    """Build a ResponseOutputItemDoneEvent-like mock."""
    from openai.types.beta.realtime import ResponseOutputItemDoneEvent
    from openai.types.beta.realtime.conversation_item import ConversationItem

    item = ConversationItem(
        type=item_type,
        call_id=call_id,
        name=name,
        arguments=arguments,
    )
    event = MagicMock(spec=ResponseOutputItemDoneEvent)
    event.type = "response.output_item.done"
    event.item = item
    event.model_dump.return_value = {"type": "response.output_item.done"}
    return event


class TestVoiceWebSocket:
    """Tests for the /ws/voice relay using a mocked OpenAI connection."""

    def _make_mock_openai_ws(self, events_to_yield):
        """
        Return a mock async context manager that yields a connection object.
        The connection iterates over `events_to_yield` and supports `.send()`.
        """
        mock_conn = MagicMock()
        mock_conn.send = AsyncMock()

        # session.update is async
        mock_session = MagicMock()
        mock_session.update = AsyncMock()
        mock_conn.session = mock_session

        async def _aiter():
            for e in events_to_yield:
                yield e

        mock_conn.__aiter__ = lambda self_: _aiter()

        # Context manager
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        return mock_ctx, mock_conn

    def test_audio_delta_forwarded_to_browser(self):
        """Audio delta events from OpenAI should be forwarded to the browser."""
        audio_event = make_event("response.audio.delta", delta=silent_pcm16_b64(256))
        mock_ctx, mock_conn = self._make_mock_openai_ws([audio_event])

        app = make_app()
        with patch("agent_server.voice_relay.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.beta.realtime.connect.return_value = mock_ctx
            mock_client_cls.return_value = mock_client

            with TestClient(app) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    msg = ws.receive_text()

        received = json.loads(msg)
        assert received["type"] == "response.audio.delta"

    def test_transcript_delta_forwarded_to_browser(self):
        """Transcript delta events should be forwarded to the browser."""
        t_event = make_event("response.audio_transcript.delta", delta="Hello!")
        mock_ctx, mock_conn = self._make_mock_openai_ws([t_event])

        app = make_app()
        with patch("agent_server.voice_relay.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.beta.realtime.connect.return_value = mock_ctx
            mock_client_cls.return_value = mock_client

            with TestClient(app) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    msg = ws.receive_text()

        received = json.loads(msg)
        assert received["type"] == "response.audio_transcript.delta"

    def test_browser_audio_forwarded_to_openai(self):
        """Audio sent by the browser should be forwarded to OpenAI."""
        # Yield one dummy event so the connection stays alive long enough
        dummy = make_event("session.created")
        mock_ctx, mock_conn = self._make_mock_openai_ws([dummy])

        app = make_app()
        with patch("agent_server.voice_relay.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.beta.realtime.connect.return_value = mock_ctx
            mock_client_cls.return_value = mock_client

            browser_event = {
                "type": "input_audio_buffer.append",
                "audio": silent_pcm16_b64(512),
            }

            with TestClient(app) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    # Drain the session.created event
                    ws.receive_text()
                    ws.send_text(json.dumps(browser_event))
                    # Disconnect so the relay tears down
                    ws.close()

        # Verify the audio event was sent to OpenAI
        sent_calls = mock_conn.send.call_args_list
        sent_types = [c.args[0].get("type") if isinstance(c.args[0], dict) else None for c in sent_calls]
        assert "input_audio_buffer.append" in sent_types

    def test_tool_call_intercepted_not_forwarded(self):
        """
        A function_call output_item.done event should be intercepted:
        - result sent back to OpenAI as conversation.item.create + response.create
        - the raw event NOT forwarded to the browser
        """
        tool_event = make_output_item_done_event(
            item_type="function_call",
            call_id="call_abc",
            name="check_return_eligibility",
            arguments=json.dumps({"days_since_purchase": 5}),
        )
        # Follow with a transcript event so the browser receives something
        transcript_event = make_event("response.audio_transcript.delta", delta="Sure!")
        mock_ctx, mock_conn = self._make_mock_openai_ws([tool_event, transcript_event])

        app = make_app()
        with patch("agent_server.voice_relay.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.beta.realtime.connect.return_value = mock_ctx
            mock_client_cls.return_value = mock_client

            with TestClient(app) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    # Only the transcript should arrive (tool event was intercepted)
                    msg = ws.receive_text()

        received = json.loads(msg)
        # The raw tool event must NOT reach the browser
        assert received["type"] != "response.output_item.done"
        # The transcript delta that followed should be the message received
        assert received["type"] == "response.audio_transcript.delta"

        # OpenAI must have received function_call_output + response.create
        sent_calls = mock_conn.send.call_args_list
        sent_types = [
            c.args[0].get("type") if isinstance(c.args[0], dict) else None
            for c in sent_calls
        ]
        assert "conversation.item.create" in sent_types
        assert "response.create" in sent_types

        # Verify the tool actually ran and produced the right output
        item_create_call = next(
            c for c in sent_calls if isinstance(c.args[0], dict) and c.args[0].get("type") == "conversation.item.create"
        )
        output = item_create_call.args[0]["item"]["output"]
        assert "Eligible" in output
        assert "25 day(s)" in output

    def test_session_update_sent_on_connect(self):
        """session.update must be called once with the right configuration."""
        mock_ctx, mock_conn = self._make_mock_openai_ws([])

        app = make_app()
        with patch("agent_server.voice_relay.AsyncOpenAI") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.beta.realtime.connect.return_value = mock_ctx
            mock_client_cls.return_value = mock_client

            with TestClient(app) as client:
                with client.websocket_connect("/ws/voice"):
                    pass

        mock_conn.session.update.assert_called_once()
        session_arg = mock_conn.session.update.call_args.kwargs["session"]
        assert session_arg["voice"] == "alloy"
        assert session_arg["turn_detection"]["type"] == "server_vad"
        tool_names = [t["name"] for t in session_arg["tools"]]
        assert "get_current_date_and_time" in tool_names
        assert "check_return_eligibility" in tool_names
