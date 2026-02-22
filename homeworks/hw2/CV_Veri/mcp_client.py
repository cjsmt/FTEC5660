from __future__ import annotations

import asyncio
import os
from typing import Dict, List

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool


MCP_BASE_URL = os.getenv("MCP_BASE_URL", "https://ftec5660.ngrok.app/mcp")

# Retry config for MCP connection (often flaky with ngrok)
MCP_LOAD_RETRIES = 3
MCP_LOAD_RETRY_DELAY = 2.0


async def load_mcp_tools() -> Dict[str, List[BaseTool]]:
    """
    Load all tools from the MCP server via MultiServerMCPClient
    and split them into LinkedIn / Facebook groups for use in graph.py.
    Retries up to MCP_LOAD_RETRIES times on connection failure.
    """
    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_BASE_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )

    last_error = None
    for attempt in range(1, MCP_LOAD_RETRIES + 1):
        try:
            tools = await client.get_tools()
            break
        except Exception as e:
            last_error = e
            if attempt < MCP_LOAD_RETRIES:
                await asyncio.sleep(MCP_LOAD_RETRY_DELAY)
            else:
                raise last_error

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