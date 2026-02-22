"""
Resume extraction subgraph: PDF path → PyPDF2 text extraction → LLM → structured ResumeData.
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
    Use PyPDF2 to extract plain text from a PDF, concatenated page by page.
    """
    from PyPDF2 import PdfReader

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def build_resume_extract_graph(llm: Any, system_prompt: str):
    """
    Build the resume-extraction subgraph: extract_pdf_text → llm_to_resume_data → END
    llm: model used to turn the PDF text into structured JSON
    system_prompt: instructions that define the ResumeData JSON format
    """

    def node_extract_pdf_text(state: ResumeExtractState) -> ResumeExtractState:
        """Extract raw text from the PDF with PyPDF2 and store it in state.ocr_text."""
        path = state.get("resume_path")
        if not path:
            return state
        try:
            raw_text = _extract_text_from_pdf(path)
            state["ocr_text"] = raw_text
            return state
        except Exception as e:
            state["ocr_text"] = f"[PDF extraction failed] {e}"
            return state

    def node_llm_to_resume_data(state: ResumeExtractState) -> ResumeExtractState:
        """Given PDF text and system_prompt, use the LLM to produce ResumeData JSON."""
        ocr_text = state.get("ocr_text") or ""
        if not ocr_text or ocr_text.startswith("["):
            return state

        user_prompt = (
            "Please read the following text extracted from a resume PDF, "
            "and extract structured information according to the required JSON format.\n\n"
            f"--- PDF raw text ---\n{ocr_text}\n\n"
            "--- Output only a single JSON object in the specified format, with no extra explanations. ---"
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
