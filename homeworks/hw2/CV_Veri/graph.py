"""
LangGraph 多智能体工作流：简历解析 -> 并行 LinkedIn/Facebook LLM 智能体（绑定 MCP 工具）-> LLM 核验报告生成。
"""

from __future__ import annotations

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from mcp_client import load_mcp_tools
from resume_extract_subgraph import build_resume_extract_graph, ResumeExtractState
from react_agent_subgraph import build_react_agent_graph, MessagesState
from utils import extract_json_from_text
from models import (
    CVState,
    FacebookProfile,
    LinkedInProfile,
    ResumeData,
)





# ===== LLM 配置 =====
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def load_prompt(name: str) -> str:
    """
    加载指定名称的 prompt 文件。
    name: 文件名（包含 .txt），如 'linkedin_agent_system.txt'
    """
    
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _build_comparison_context(resume: Optional[ResumeData], linkedin: Optional[LinkedInProfile], facebook: Optional[FacebookProfile]) -> str:
    """将 resume、LinkedIn、Facebook 信息序列化为 LLM 可读的上下文。"""
    parts = []
    parts.append("## 简历信息 (Resume)")
    parts.append(json.dumps(resume, indent=2, ensure_ascii=False) if resume else "无")
    parts.append("\n## LinkedIn 档案")
    parts.append(json.dumps(linkedin, indent=2, ensure_ascii=False) if linkedin else "未找到")
    parts.append("\n## Facebook 档案")
    parts.append(json.dumps(facebook, indent=2, ensure_ascii=False) if facebook else "未找到")
    return "\n".join(parts)



def build_graph():
    """
    构建 LangGraph 工作流：

    START -> extract_resume -> fetch_social_profiles -> compare_and_report -> END

    - extract_resume: 解析 PDF 简历
    - fetch_social_profiles: LinkedIn 与 Facebook 两个 LLM 智能体并行运行，各自绑定 MCP 工具
    - compare_and_report: LLM 智能体分析对比并生成 VerificationReport

    LLM 实例与 agent 在 build_graph 中集中创建，通过闭包传入各节点，避免重复创建。
    """
    llm = init_chat_model(
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_provider=os.getenv("LLM_PROVIDER"),
    )

    def node_extract_resume(state: CVState) -> CVState:
        """节点 1：子图执行 PDF 原文提取 → LLM 结构化，得到 ResumeData。"""
        resume_path = state["resume_path"]
        resume_prompt = load_prompt("resume_extract_system.txt")
        extract_app = build_resume_extract_graph(llm, resume_prompt)
        state0: ResumeExtractState = {"resume_path": resume_path}
        result = extract_app.invoke(state0)
        state["resume_data"] = result.get("resume_data")
        return state


    tool_sets = asyncio.run(load_mcp_tools())
    linkedin_tools: List[BaseTool] = tool_sets["linkedin"]
    facebook_tools: List[BaseTool] = tool_sets["facebook"]


    def node_fetch_social_profiles(state: CVState) -> CVState:
        """节点 2：LinkedIn 与 Facebook 两个 LLM 智能体并行执行。"""
        resume_data = state.get("resume_data")
        if not resume_data or not resume_data.get("name"):
            return state

        name = resume_data["name"]
        city = resume_data.get("city") or ""
        country = resume_data.get("country") or ""
        system_prompt_li = load_prompt("linkedin_agent_system.txt")
        system_prompt_fb = load_prompt("facebook_agent_system.txt")
        user_prompt_li = (
            "请根据以下简历信息查找并获取该候选人的 LinkedIn 档案：\n\n"
            f"姓名：{name}\n"
            f"城市：{city or '未提供'}\n"
            f"国家：{country or '未提供'}\n"
        )
        user_prompt_fb = (
            "请根据以下简历信息查找并获取该候选人的 Facebook 档案：\n\n"
            f"姓名：{name}\n"
            f"城市：{city or '未提供'}\n"
            f"国家：{country or '未提供'}\n"
        )

        linkedin_profile = None
        facebook_profile = None

        async def run_agent_once(tools, system_prompt, user_prompt):
            # 构建一次性的 ReAct 子图
            agent_app = build_react_agent_graph(llm, tools, system_prompt)
            # 初始 state 只有 messages
            state0: MessagesState = {
                "messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            }
            # 增加循环轮数到10轮，以支持多候选人查询场景
            current_state = state0
            for _ in range(10):
                current_state = await agent_app.ainvoke(current_state)
                msgs = current_state.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage) and not getattr(msgs[-1], "tool_calls", None):
                    # 最后一条 AIMessage 没有 tool_calls，认为结束
                    break
            return current_state.get("messages", [])
            # result = agent_app.invoke({
            #     "messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            # })
            # return result.get("messages", [])

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_li = executor.submit(asyncio.run, run_agent_once(linkedin_tools, system_prompt_li, user_prompt_li))
            future_fb = executor.submit(asyncio.run, run_agent_once(facebook_tools, system_prompt_fb, user_prompt_fb))

            for future in as_completed([future_li, future_fb]):
                msgs = future.result()
                
                profile_data = None
                is_linkedin = (future is future_li)
                
                # 1. 优先从最后一条 AIMessage 的文本中提取 JSON（Agent 应该在这里输出最终格式的 JSON）
                last_ai = next(
                    (m for m in reversed(msgs) if isinstance(m, AIMessage)), None
                )
                
                # 检查最后一条 AIMessage 是否还有 tool_calls（说明 Agent 可能还没完成）
                if last_ai and not getattr(last_ai, "tool_calls", None) and last_ai.content:
                    # Agent 已经完成，从文本中提取 JSON
                    data = extract_json_from_text(
                        last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
                    )
                    if data and not data.get("found") is False and not data.get("error"):
                        profile_data = data
                
                # 2. 如果从 AIMessage 中提取不到有效数据，尝试从 ToolMessage 中提取最后一个 get_linkedin_profile/get_facebook_profile 的结果
                # （这种情况可能发生在 Agent 在10轮内还没生成最终 JSON，但工具已经返回了数据）
                if not profile_data:
                    tool_name = "get_linkedin_profile" if is_linkedin else "get_facebook_profile"
                    # 从后往前找最后一个目标工具的 ToolMessage
                    for i in range(len(msgs) - 1, -1, -1):
                        msg = msgs[i]
                        if isinstance(msg, ToolMessage) and i > 0:
                            prev_msg = msgs[i - 1]
                            if isinstance(prev_msg, AIMessage) and getattr(prev_msg, "tool_calls", None):
                                for tc in prev_msg.tool_calls:
                                    tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
                                    if tc_name == tool_name:
                                        # 找到目标工具的返回结果，尝试解析 JSON
                                        try:
                                            tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                                            if isinstance(tool_result, dict) and not tool_result.get("found") is False:
                                                # 使用工具返回的原始数据（Agent 应该会进一步处理，但这里作为备选）
                                                profile_data = tool_result
                                                break
                                        except (json.JSONDecodeError, AttributeError):
                                            pass
                                if profile_data:
                                    break
                
                # 3. 如果找到了 profile 数据，保存到 state
                if profile_data and not profile_data.get("found") is False and not profile_data.get("error"):
                    if is_linkedin:
                        linkedin_profile = profile_data
                    else:
                        facebook_profile = profile_data   

        state["linkedin_profile"] = linkedin_profile
        state["facebook_profile"] = facebook_profile
        return state

    def node_compare_and_report(state: CVState) -> CVState:
        """节点 3：LLM 智能体分析对比并生成结构化验证报告。"""
        resume = state.get("resume_data")
        linkedin = state.get("linkedin_profile")
        facebook = state.get("facebook_profile")

        if resume is None:
            state["report"] = {
                "resume": {},
                "summary": "未能成功解析简历内容。",
            }
            return state

        system_prompt = load_prompt("compare_and_report_system.txt")
        context = _build_comparison_context(resume, linkedin, facebook)
        user_prompt = f"请根据以下信息生成验证报告：\n\n{context}"

        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            
            # 从响应中提取 JSON
            analysis_data = extract_json_from_text(content)
            if not analysis_data:
                raise ValueError("未能从 LLM 响应中提取 JSON")

            sk = analysis_data.get("skills_comparison") or {}
            ex = analysis_data.get("experience_comparison") or {}
            ed = analysis_data.get("education_comparison") or {}
            s1 = sk.get("score") if isinstance(sk.get("score"), (int, float)) else 0.0
            s2 = ex.get("score") if isinstance(ex.get("score"), (int, float)) else 0.0
            s3 = ed.get("score") if isinstance(ed.get("score"), (int, float)) else 0.0
            average_score = (s1 + s2 + s3) / 3.0

            state["report"] = {
                "resume": resume,
                "linkedin_profile": linkedin,
                "facebook_profile": facebook,
                "skills_comparison": analysis_data.get("skills_comparison"),
                "experience_comparison": analysis_data.get("experience_comparison"),
                "education_comparison": analysis_data.get("education_comparison"),
                "summary": analysis_data.get("summary", "LLM 生成验证报告时出错。"),
                "average_score": round(average_score, 4),
            }
        except Exception:
            state["report"] = {
                "resume": resume,
                "linkedin_profile": linkedin,
                "facebook_profile": facebook,
                "summary": "LLM 生成验证报告时出错，请检查 API 配置。",
                "average_score": None,
            }

        return state

    builder = StateGraph(CVState)

    builder.add_node("extract_resume", node_extract_resume)
    builder.add_node("fetch_social_profiles", node_fetch_social_profiles)
    builder.add_node("compare_and_report", node_compare_and_report)

    builder.add_edge(START, "extract_resume")
    builder.add_edge("extract_resume", "fetch_social_profiles")
    builder.add_edge("fetch_social_profiles", "compare_and_report")
    builder.add_edge("compare_and_report", END)

    return builder.compile()
