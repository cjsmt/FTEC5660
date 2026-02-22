import json
import re
import ast
from typing import Any, Dict, Optional

from models import ResumeData, LinkedInProfile, FacebookProfile

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from an agent's text response.
    Supports plain JSON or JSON wrapped in ```json ... ``` fences.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    # Try to extract the content inside ```json ... ``` or generic ``` ... ``` fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None



def extract_json_str_from_content(content: Any) -> Optional[str]:
    """
    Extract a JSON string from MCP-style content.
    Content can be: (1) list like [{"type":"text","text":"{...}"}], (2) str that is Python repr of such list,
    or (3) plain JSON string.
    """
    if content is None:
        return None
    if isinstance(content, list) and content:
        block = content[0]
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    if isinstance(content, str):
        s = content.strip()
        if not s:
            return None
        if s.startswith("[") or s.startswith("{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "text" in parsed[0]:
                    return parsed[0]["text"]
                if isinstance(parsed, dict):
                    return s
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "text" in parsed[0]:
                        return parsed[0]["text"]
                except (ValueError, SyntaxError):
                    pass
        return s
    return None


def build_comparison_context(resume: Optional[ResumeData], linkedin: Optional[LinkedInProfile], facebook: Optional[FacebookProfile]) -> str:
    """Serialize resume, LinkedIn and Facebook data into an LLM-readable context string."""
    parts = []
    parts.append("## Resume data")
    parts.append(json.dumps(resume, indent=2, ensure_ascii=False) if resume else "N/A")
    parts.append("\n## LinkedIn profile")
    parts.append(json.dumps(linkedin, indent=2, ensure_ascii=False) if linkedin else "Not found")
    parts.append("\n## Facebook profile")
    parts.append(json.dumps(facebook, indent=2, ensure_ascii=False) if facebook else "Not found")
    return "\n".join(parts)