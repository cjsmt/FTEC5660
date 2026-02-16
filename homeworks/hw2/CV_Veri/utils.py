import json
import re
from typing import Any, Dict, Optional

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从 agent 的文本回复中提取 JSON。支持纯 JSON 或 ```json ... ``` 包裹的形式。
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    # 尝试提取 ```json ... ``` 或 ``` ... ``` 中的内容
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None