import litellm
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    get_session_id,
    process_agent_stream_events,
    sanitize_output_items,
)

# NOTE: this will work for all databricks models OTHER than GPT-OSS, which uses a slightly different API
set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

CUSTOMER_SUPPORT_INSTRUCTIONS = """You are a friendly and helpful customer support agent for Acme Corp.

Your role is to assist customers with their questions, concerns, and issues. You should:

1. Greet customers warmly and acknowledge their concern.
2. Listen carefully to understand their issue fully before responding.
3. Provide clear, accurate information about products, services, policies, and procedures.
4. Troubleshoot issues step-by-step when customers have technical problems.
5. Escalate appropriately - if you cannot resolve an issue, let the customer know you will connect them with a specialist.
6. Be empathetic - validate customer frustrations and show you care about resolving their issue.
7. Confirm resolution - always check if the customer's issue has been fully addressed before ending the conversation.

Common topics you can help with:
- Order status and tracking
- Returns and refunds (standard policy: 30-day return window, full refund)
- Product information and recommendations
- Account management (password reset, profile updates)
- Billing and payment questions
- Shipping and delivery inquiries
- Technical support for Acme products

Always be professional, patient, and solution-oriented. If you don't know something, be honest and offer to find out.
"""


@function_tool
def get_current_date_and_time() -> str:
    """Returns the current date and time in UTC. Use this when a customer asks about
    time-sensitive matters such as return windows, order cutoffs, or business hours."""
    return datetime.now(timezone.utc).strftime("%A, %B %d, %Y at %H:%M UTC")


@function_tool
def check_return_eligibility(days_since_purchase: int) -> str:
    """Checks whether a purchase is eligible for return based on the number of days
    since purchase. Acme Corp has a 30-day return policy.

    Args:
        days_since_purchase: Number of days since the customer made the purchase.
    """
    if days_since_purchase <= 30:
        remaining = 30 - days_since_purchase
        return f"Eligible for return. Customer has {remaining} day(s) remaining in their return window."
    else:
        return f"Not eligible for return. The 30-day return window has passed ({days_since_purchase} days since purchase)."


def create_customer_support_agent() -> Agent:
    return Agent(
        name="Customer Support Agent",
        instructions=CUSTOMER_SUPPORT_INSTRUCTIONS,
        model="databricks-gpt-5-2",
        tools=[get_current_date_and_time, check_return_eligibility],
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    agent = create_customer_support_agent()
    messages = [i.model_dump() for i in request.input]
    result = await Runner.run(agent, messages)
    return ResponsesAgentResponse(output=sanitize_output_items(result.new_items))


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    agent = create_customer_support_agent()
    messages = [i.model_dump() for i in request.input]
    result = Runner.run_streamed(agent, input=messages)

    async for event in process_agent_stream_events(result.stream_events()):
        yield event
