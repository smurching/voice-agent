"""
Voice relay: browser WebSocket <-> OpenAI Realtime API.

Browser streams PCM16 audio over WebSocket /ws/voice.
This relay forwards it to OpenAI's Realtime API, intercepts tool calls to
execute them server-side using existing Python functions, and relays audio
and transcript events back to the browser.

Deployment note: OPENAI_API_KEY must be set in the environment.
For Databricks Apps, add it as a secret and reference it in app.yaml.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from openai.types.beta.realtime import ResponseOutputItemDoneEvent
from openai.types.beta.realtime.session_update_event_param import Session, SessionTool

from agent_server.agent import CUSTOMER_SUPPORT_INSTRUCTIONS

router = APIRouter()

TEMPLATES_DIR = Path(__file__).parent / "templates"

TOOLS_SCHEMA: list[SessionTool] = [
    SessionTool(
        type="function",
        name="get_current_date_and_time",
        description=(
            "Returns the current date and time in UTC. Use this when a customer asks about "
            "time-sensitive matters such as return windows, order cutoffs, or business hours."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    SessionTool(
        type="function",
        name="check_return_eligibility",
        description=(
            "Checks whether a purchase is eligible for return based on the number of days "
            "since purchase. Acme Corp has a 30-day return policy."
        ),
        parameters={
            "type": "object",
            "properties": {
                "days_since_purchase": {
                    "type": "integer",
                    "description": "Number of days since the customer made the purchase.",
                }
            },
            "required": ["days_since_purchase"],
        },
    ),
]


def _execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return the string result."""
    if name == "get_current_date_and_time":
        return datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")
    elif name == "check_return_eligibility":
        days = int(arguments.get("days_since_purchase", 0))
        if days <= 30:
            remaining = 30 - days
            return f"Eligible for return. Customer has {remaining} day(s) remaining in their return window."
        else:
            return f"Not eligible for return. The 30-day return window has passed ({days} days since purchase)."
    else:
        return f"Unknown tool: {name}"


@router.get("/voice")
async def voice_ui():
    return FileResponse(TEMPLATES_DIR / "voice.html")


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as openai_ws:
        await openai_ws.session.update(
            session=Session(
                instructions=CUSTOMER_SUPPORT_INSTRUCTIONS,
                voice="alloy",
                turn_detection={"type": "server_vad"},
                input_audio_format="pcm16",
                output_audio_format="pcm16",
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )
        )

        async def browser_to_openai():
            """Forward messages from browser to OpenAI."""
            try:
                while True:
                    data = await websocket.receive_text()
                    event = json.loads(data)
                    await openai_ws.send(event)
            except WebSocketDisconnect:
                pass

        async def openai_to_browser():
            """Forward events from OpenAI to browser, intercepting tool calls."""
            try:
                async for event in openai_ws:
                    # Intercept completed function_call output items to execute tools
                    if (
                        isinstance(event, ResponseOutputItemDoneEvent)
                        and event.item.type == "function_call"
                    ):
                        item = event.item
                        call_id = item.call_id or ""
                        name = item.name or ""
                        try:
                            arguments = json.loads(item.arguments) if item.arguments else {}
                        except json.JSONDecodeError:
                            arguments = {}

                        result = _execute_tool(name, arguments)

                        await openai_ws.send({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": result,
                            },
                        })
                        await openai_ws.send({"type": "response.create"})
                        # Don't forward the raw tool call event to the browser
                        continue

                    await websocket.send_text(json.dumps(event.model_dump()))

            except WebSocketDisconnect:
                pass

        await asyncio.gather(browser_to_openai(), openai_to_browser())
