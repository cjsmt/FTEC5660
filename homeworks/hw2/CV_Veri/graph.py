"""
LangGraph multi-agent workflow:
PDF resume parsing → parallel LinkedIn/Facebook LLM agents (backed by MCP tools) → LLM-generated verification report.
"""

from __future__ import annotations

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from mcp_client import load_mcp_tools
from resume_extract_subgraph import build_resume_extract_graph, ResumeExtractState
from react_agent_subgraph import build_react_agent_graph, MessagesState
from utils import extract_json_from_text, extract_json_str_from_content, build_comparison_context
from models import CVState


# ===== LLM configuration =====
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

def load_prompt(name: str) -> str:
    """
    Load a prompt file by name.
    name: filename (with .txt), e.g. 'linkedin_agent_system.txt'
    """
    
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_graph():
    """
    Build the LangGraph workflow:

    START -> extract_resume -> fetch_social_profiles -> compare_and_report -> END

    - extract_resume: parse the PDF resume
    - fetch_social_profiles: run LinkedIn and Facebook LLM agents in parallel, each bound to MCP tools
    - compare_and_report: use an LLM to compare and generate a structured VerificationReport

    The LLM instance and agents are created once in build_graph and passed into nodes via closures
    to avoid redundant construction.
    """
    llm = init_chat_model(
        model=os.getenv("LLM_MODEL_NAME"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model_provider=os.getenv("LLM_PROVIDER"),
    )

    def node_extract_resume(state: CVState) -> CVState:
        """Node 1: run the resume-extraction subgraph (PDF text → LLM → ResumeData)."""
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
        """Node 2: run LinkedIn and Facebook LLM agents in parallel."""
        resume_data = state.get("resume_data")
        if not resume_data or not resume_data.get("name"):
            return state

        name = resume_data["name"]
        city = resume_data.get("city") or ""
        country = resume_data.get("country") or ""
        system_prompt_li = load_prompt("linkedin_agent_system.txt")
        system_prompt_fb = load_prompt("facebook_agent_system.txt")
        user_prompt_li = (
            "Please use the following resume information to search for and retrieve the candidate's LinkedIn profile.\n\n"
            f"Name: {name}\n"
            f"City: {city or 'not provided'}\n"
            f"Country: {country or 'not provided'}\n"
        )
        user_prompt_fb = (
            "Please use the following resume information to search for and retrieve the candidate's Facebook profile.\n\n"
            f"Name: {name}\n"
            f"City: {city or 'not provided'}\n"
            f"Country: {country or 'not provided'}\n"
        )

        linkedin_profile = None
        facebook_profile = None

        async def run_agent_once(tools, system_prompt, user_prompt):
            # Build a one-off ReAct subgraph for a single agent run
            agent_app = build_react_agent_graph(llm, tools, system_prompt)
            # Initial state only contains messages
            state0: MessagesState = {
                "messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            }
            # Allow up to 10 iterations to support multi-candidate search scenarios
            current_state = state0
            for _ in range(10):
                current_state = await agent_app.ainvoke(current_state)
                msgs = current_state.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage) and not getattr(msgs[-1], "tool_calls", None):
                    # If the last AIMessage has no tool_calls, treat this as completion
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
                is_linkedin = (future is future_li)
                platform = "LinkedIn" if is_linkedin else "Facebook"

                profile_data = None
                
                # 1. Prefer extracting JSON from the last AIMessage (the agent should output final JSON here)
                last_ai = next(
                    (m for m in reversed(msgs) if isinstance(m, AIMessage)), None
                )
                
                # Check whether the last AIMessage still has tool_calls (agent may not be finished yet)
                if last_ai and not getattr(last_ai, "tool_calls", None) and last_ai.content:
                    # Agent has finished; extract JSON from the text
                    raw_str = extract_json_str_from_content(last_ai.content)
                    data = extract_json_from_text(raw_str) if raw_str else None
                    if data and not data.get("found") is False and not data.get("error"):
                        profile_data = data
                
                # 2. If we cannot get valid data from AIMessage, fall back to the last
                #    ToolMessage for get_linkedin_profile/get_facebook_profile.
                #    This covers the case where tools have already returned data but the
                #    agent has not produced a final JSON within the iteration limit.
                if not profile_data:
                    tool_name = "get_linkedin_profile" if is_linkedin else "get_facebook_profile"
                    # Scan backwards to find the last ToolMessage for the target tool
                    for i in range(len(msgs) - 1, -1, -1):
                        msg = msgs[i]
                        if isinstance(msg, ToolMessage) and i > 0:
                            prev_msg = msgs[i - 1]
                            if isinstance(prev_msg, AIMessage) and getattr(prev_msg, "tool_calls", None):
                                for tc in prev_msg.tool_calls:
                                    tc_name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
                                    if tc_name == tool_name:
                                        # Found the target tool result; extract JSON from MCP-style content
                                        try:
                                            json_str = extract_json_str_from_content(msg.content)
                                            tool_result = json.loads(json_str) if json_str else None
                                            if not isinstance(tool_result, dict):
                                                tool_result = extract_json_from_text(json_str) if json_str else None
                                            if isinstance(tool_result, dict) and not tool_result.get("found") is False:
                                                # Use the raw tool result as a fallback (agent normally post-processes it)
                                                profile_data = tool_result
                                                break
                                        except (json.JSONDecodeError, AttributeError):
                                            pass
                                if profile_data:
                                    break
                
                # 3. If we found a valid profile, store it in the state
                if profile_data and not profile_data.get("found") is False and not profile_data.get("error"):
                    if is_linkedin:
                        linkedin_profile = profile_data
                    else:
                        facebook_profile = profile_data

        state["linkedin_profile"] = linkedin_profile
        state["facebook_profile"] = facebook_profile
        return state

    def node_compare_and_report(state: CVState) -> CVState:
        """Node 3: LLM compares data and generates a structured verification report."""
        resume = state.get("resume_data")
        linkedin = state.get("linkedin_profile")
        facebook = state.get("facebook_profile")

        if resume is None:
            state["report"] = {
                "resume": {},
                "summary": "Failed to parse resume content.",
            }
            return state

        system_prompt = load_prompt("compare_and_report_system.txt")
        context = build_comparison_context(resume, linkedin, facebook)
        user_prompt = f"Please generate a verification report based on the following information:\n\n{context}"

        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            
            # Extract JSON from the LLM response
            analysis_data = extract_json_from_text(content)
            if not analysis_data:
                raise ValueError("Could not extract JSON from LLM response")

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
                "summary": analysis_data.get("summary", "Error while generating verification report with LLM."),
                "average_score": round(average_score, 4),
            }
        except Exception:
            state["report"] = {
                "resume": resume,
                "linkedin_profile": linkedin,
                "facebook_profile": facebook,
                "summary": "Error while generating verification report with LLM. Please check API configuration.",
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
