from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool


class MessagesState(TypedDict, total=False):
    messages: List[Any]


def build_react_agent_graph(llm: Any, tools: List[BaseTool], system_prompt: str):
    """
    构建一个最小 ReAct 子图：
    - state: {"messages": [...]}
    - llm 节点：根据 messages 生成下一条 AIMessage（可带 tool_calls）
    - tool 节点：根据最近一次 AIMessage 中的 tool_calls 调用 MCP 工具，并把 ToolMessage 加回 messages
    在外层通过重复调用 llm_node -> tool_node 的方式实现多轮 ReAct。
    """
    llm_with_tools = llm.bind_tools(tools)

    async def llm_node(state: MessagesState) -> MessagesState:
        messages = state.get("messages", [])
        if not messages:
            messages = [SystemMessage(content=system_prompt)]
        # LLM 只看当前 messages
        ai_message = await llm_with_tools.ainvoke(messages)  # 这里用同步 invoke，llm 本身可以是异步封装
        messages.append(ai_message)
        return {"messages": messages}

    async def tool_node(state: MessagesState) -> MessagesState:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": messages}

        last = messages[-1]
        if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
            # 没有 tool_calls，直接结束
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
            new_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call_id,
                )
            )

        return {"messages": new_messages}

    builder = StateGraph(MessagesState)
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_edge("llm", "tools")
    # 不直接连到 END，由外层控制迭代次数和终止条件
    # builder.add_edge("tools", "llm")

    # 注意：这里不 compile 成带 END 的完整图，而是留给外层控制流
    return builder.compile()