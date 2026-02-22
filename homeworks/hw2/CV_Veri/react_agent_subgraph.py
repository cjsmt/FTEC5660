from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool


class MessagesState(TypedDict, total=False):
    messages: List[Any]


def build_react_agent_graph(llm: Any, tools: List[BaseTool], system_prompt: str):
    """
    Build a minimal ReAct-style subgraph:
    - state: {\"messages\": [...]}
    - llm node: generate the next AIMessage from messages (optionally with tool_calls)
    - tool node: execute tool_calls from the latest AIMessage and append ToolMessages to messages
    The outer controller calls llm_node -> tool_node repeatedly to realize multi-step ReAct behavior.
    """
    llm_with_tools = llm.bind_tools(tools)

    async def llm_node(state: MessagesState) -> MessagesState:
        messages = state.get("messages", [])
        if not messages:
            messages = [SystemMessage(content=system_prompt)]
        # LLM only sees the current messages
        ai_message = await llm_with_tools.ainvoke(messages)
        messages.append(ai_message)
        return {"messages": messages}

    async def tool_node(state: MessagesState) -> MessagesState:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": messages}

        last = messages[-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            # No tool_calls -> nothing to do here
            return {"messages": messages}

        tool_by_name = {t.name: t for t in tools}
        new_messages = messages[:]
        for tc in last.tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
            if not name or name not in tool_by_name:
                continue
            args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else None)
            if isinstance(args, str):
                try:
                    import json
                    args = json.loads(args)
                except Exception:
                    args = {"input": args}
            tool = tool_by_name[name]
            result = await tool.ainvoke(args)
            tool_call_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
            content = result
            if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                content = content[0]["text"]
            elif not isinstance(content, str):
                content = str(content)
            new_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
            )

        return {"messages": new_messages}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_edge("llm", "tools")
    # We intentionally do not connect \"tools\" back to END here; the outer controller
    # is responsible for iterating and deciding termination conditions.

    return builder.compile()