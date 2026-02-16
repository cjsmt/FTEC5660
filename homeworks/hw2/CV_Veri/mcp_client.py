from __future__ import annotations

import os
from typing import Dict, List, Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
import asyncio


MCP_BASE_URL = os.getenv("MCP_BASE_URL", "https://ftec5660.ngrok.app/mcp")


async def load_mcp_tools() -> Dict[str, List[BaseTool]]:
    """
    通过 MultiServerMCPClient 从 MCP 服务器加载所有工具，
    并按名字拆成 LinkedIn / Facebook 两组，供 graph.py 使用。
    """
    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_BASE_URL,
                "headers": {"ngrok-skip-browser-warning": "true"}
            }
        }
    )

    tools = await client.get_tools()  # List[BaseTool]

    by_name = {t.name: t for t in tools}

    linkedin_tools = [
        by_name[name]
        for name in [
            "search_linkedin_people",
            "get_linkedin_profile",
            "get_linkedin_interactions",
        ]
        if name in by_name
    ]

    facebook_tools = [
        by_name[name]
        for name in [
            "search_facebook_users",
            "get_facebook_profile",
            "get_facebook_mutual_friends",
        ]
        if name in by_name
    ]

    return {
        "linkedin": linkedin_tools,
        "facebook": facebook_tools,
    }