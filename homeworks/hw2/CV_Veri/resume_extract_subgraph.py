"""
简历提取子图：PDF 路径 → PyPDF2 提取原文 → LLM 结构化为 ResumeData。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from models import ResumeData
from utils import extract_json_from_text


class ResumeExtractState(TypedDict, total=False):
    resume_path: str
    ocr_text: str
    resume_data: Optional[ResumeData]


def _extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    使用 PyPDF2 从 PDF 提取纯文本，按页拼接。
    """
    from PyPDF2 import PdfReader

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def build_resume_extract_graph(llm: Any, system_prompt: str):
    """
    构建简历提取子图：extract_pdf_text → llm_to_resume_data → END
    llm: 用于将 PDF 原文结构化为 JSON 的模型
    system_prompt: 规定输出为 ResumeData JSON 的系统提示
    """

    def node_extract_pdf_text(state: ResumeExtractState) -> ResumeExtractState:
        """用 PyPDF2 从 PDF 提取原文，写入 state.ocr_text。"""
        path = state.get("resume_path")
        if not path:
            return state
        try:
            raw_text = _extract_text_from_pdf(path)
            state["ocr_text"] = raw_text
            return state
        except Exception as e:
            state["ocr_text"] = f"[PDF 提取失败] {e}"
            return state

    def node_llm_to_resume_data(state: ResumeExtractState) -> ResumeExtractState:
        """根据 PDF 原文与 system_prompt，用 LLM 生成并解析为 ResumeData JSON。"""
        ocr_text = state.get("ocr_text") or ""
        if not ocr_text or ocr_text.startswith("["):
            return state

        user_prompt = (
            "请根据以下从简历 PDF 中提取的原文，抽取结构化信息并输出符合规定格式的 JSON。\n\n"
            f"--- PDF 原文 ---\n{ocr_text}\n\n"
            "--- 请只输出上述格式的 JSON，不要其他说明 ---"
        )

        try:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
            )
            content = response.content if hasattr(response, "content") else str(response)
            data = extract_json_from_text(content)
            if data and isinstance(data, dict):
                resume_data: ResumeData = {
                    "name": data.get("name"),
                    "city": data.get("city"),
                    "country": data.get("country"),
                    "hometown": data.get("hometown"),
                    "headline": data.get("headline"),
                    "skills": data.get("skills") if isinstance(data.get("skills"), list) else [],
                    "experience": data.get("experience") if isinstance(data.get("experience"), list) else [],
                    "education": data.get("education") if isinstance(data.get("education"), list) else [],
                    "raw_text": ocr_text,
                }
                return {**state, "resume_data": resume_data}
        except Exception:
            pass
        return state

    builder = StateGraph(ResumeExtractState)
    builder.add_node("extract_pdf_text", node_extract_pdf_text)
    builder.add_node("llm_to_resume", node_llm_to_resume_data)

    builder.add_edge(START, "extract_pdf_text")
    builder.add_edge("extract_pdf_text", "llm_to_resume")
    builder.add_edge("llm_to_resume", END)

    return builder.compile()
