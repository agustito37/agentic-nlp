"""Pizza-ordering agent built with LangGraph"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


PIZZA_PRICES: dict[str, float] = {
    "cheese": 8.0,
    "pepperoni": 10.0,
    "margherita": 9.0,
    "hawaiian": 11.0,
    "vegetarian": 10.0,
    "vegan": 12.0,
    "gluten-free": 13.0,
    "bbq-chicken": 12.0,
    "buffalo-chicken": 12.0,
}

PIZZA_LIST = ", ".join(PIZZA_PRICES.keys())

SYSTEM_PROMPT = (
    '''You are a friendly pizza-ordering assistant.
    - If the user greets you, greet back and mention you can help order pizza.
    - If the user asks what pizzas are available or for the menu, call the 
    `list_pizza_types` tool.
    - If the user wants to order a pizza, call the `confirm_pizza_order` tool 
    with the pizza type and quantity (default quantity is 1). 
    Normalize the pizza type to lowercase with dashes 
    (e.g. 'BBQ Chicken' -> 'bbq-chicken', 'gluten free' -> 'gluten-free'). 
    Convert spelled-out numbers like 'two' to 2.
    - If the user says goodbye, reply: 
    'Bye, thanks for using the pizza ordering service.'
    - Keep replies short and friendly.'''
)


@tool
def list_pizza_types() -> str:
    """Return the list of available pizza types."""
    return f"We have these pizza types: {PIZZA_LIST}."


@tool
def confirm_pizza_order(pizza_type: str, quantity: int = 1) -> str:
    """Confirm a pizza order and return the total price.

    Args:
        pizza_type: One of the available pizza types
            (cheese, pepperoni, margherita, hawaiian, vegetarian, vegan,
             gluten-free, bbq-chicken, buffalo-chicken).
        quantity: Number of pizzas (defaults to 1).
    """
    normalized = pizza_type.strip().lower().replace(" ", "-").replace("_", "-")
    if normalized not in PIZZA_PRICES:
        return "Sorry, I can only order pizzas from the available menu."

    if quantity is None or quantity < 1:
        quantity = 1

    unit_price = PIZZA_PRICES[normalized]
    total = unit_price * quantity
    plural = "pizza" if quantity == 1 else "pizzas"
    return (
        f"Great choice. I can help you order {quantity} {normalized} "
        f"{plural}. The total is ${total:.2f}."
    )


TOOLS = [list_pizza_types, confirm_pizza_order]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def build_agent():
    """Build the compiled LangGraph agent with in-memory checkpointing."""
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).bind_tools(TOOLS)

    def chatbot(state: AgentState) -> AgentState:
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm.invoke(messages)]}

    graph = StateGraph(AgentState)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("chatbot")
    graph.add_conditional_edges(
        "chatbot",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "chatbot")

    return graph.compile(checkpointer=MemorySaver())


def chat(agent, user_message: str, thread_id: str = "default") -> str:
    """Send a user message to the agent and return the assistant reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Pizza agent ready. Type 'quit' to exit.")
    agent = build_agent()
    thread = "cli"
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            break
        reply = chat(agent, user, thread_id=thread)
        print(f"Bot: {reply}")
