"""Tool dispatch for the pipeline agents.

Agents don't import tools directly; they call

    invoke_tool_sync("business_validation", {"translation_result": ...})

and this module finds the tool and runs it in-process.

Each tool is imported only when it is first asked for, so an agent that
only needs one tool doesn't fail because some other tool has a missing
dependency (for example qdrant-client for the RAG tools).
"""

from __future__ import annotations

import importlib
from functools import lru_cache

from langchain_core.tools import BaseTool

# tool name -> module inside backend.tools (the tool object is "<name>_tool")
TOOL_MODULES = {
    "invoice_watcher": "backend.tools.invoice_watcher_tool",
    "data_harvester": "backend.tools.data_harvester_tool",
    "lang_bridge": "backend.tools.lang_bridge_tool",
    "data_completeness_checker": "backend.tools.data_completeness_checker_tool",
    "business_validation": "backend.tools.business_validation_tool",
    "insight_reporter": "backend.tools.insight_reporter_tool",
    "vector_indexer": "backend.tools.rag_tools.vector_indexer_tool",
    "semantic_retriever": "backend.tools.rag_tools.semantic_retriever_tool",
    "chunk_ranker": "backend.tools.rag_tools.chunk_ranker_tool",
    "response_synthesizer": "backend.tools.rag_tools.response_synthesizer_tool",
}


@lru_cache(maxsize=None)
def get_tool_by_name(tool_name: str, use_cache: bool = True) -> BaseTool:
    """Import and return one tool by its short name, e.g. 'lang_bridge'."""
    if tool_name not in TOOL_MODULES:
        raise ValueError(f"Unknown local tool: {tool_name}. Known tools: {', '.join(TOOL_MODULES)}")

    module = importlib.import_module(TOOL_MODULES[tool_name])
    tool = getattr(module, f"{tool_name}_tool", None)
    if not isinstance(tool, BaseTool):
        raise ValueError(f"{TOOL_MODULES[tool_name]} has no LangChain tool named {tool_name}_tool")
    return tool


def get_tools_sync(use_cache: bool = True) -> list[BaseTool]:
    """All tools (imports every tool module)."""
    return [get_tool_by_name(name) for name in TOOL_MODULES]


def invoke_tool_sync(tool_name: str, tool_input: dict, use_cache: bool = True):
    """Run a tool with its input dict and return what it returns."""
    return get_tool_by_name(tool_name).invoke(tool_input)