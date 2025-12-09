import os
from typing import Any, Dict, List

from mcp_client.client import MCPClient


# Simple wrappers around MCP tools so we can register them as tools for an agent.
def _get_client(client: MCPClient | None) -> MCPClient:
    return client or MCPClient()


def tool_retrieve(question: str, top_k: int = 5, file_ids: List[str] | None = None,
                  client: MCPClient | None = None) -> str:
    """
    Retrieve relevant chunks from the PDF store. Use when the question is about the uploaded PDF.
    Returns a concatenated text context.
    """
    mcp = _get_client(client)
    chunks: List[str] = mcp.invoke(
        "retrieve_chunks",
        {"question": question, "top_k": top_k, "file_ids": file_ids or []},
    )
    return "\n\n".join(chunks)


def tool_web_search(query: str, num_results: int = 5, client: MCPClient | None = None) -> str:
    """
    Perform web search when PDF retrieval is empty or not relevant. Returns snippets joined by newline.
    """
    mcp = _get_client(client)
    results: List[str] = mcp.invoke("web_search_tool", {"query": query, "num_results": num_results})
    return "\n".join(results)


def tool_memory_get(session_id: str, max_rows: int = 5, client: MCPClient | None = None) -> str:
    """
    Fetch recent conversation history for the given session.
    """
    mcp = _get_client(client)
    lines: List[str] = mcp.invoke("memory_get", {"session_id": session_id, "max_rows": max_rows})
    return "\n".join(lines)


def tool_memory_add(session_id: str, query: str, answer: str, chunk_index: int | None = None,
                    client: MCPClient | None = None) -> str:
    """
    Persist Q/A into history. Call after generating the final answer.
    """
    mcp = _get_client(client)
    return mcp.invoke(
        "memory_add",
        {
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "chunk_index": chunk_index,
        },
    )


def tool_compare_pdfs(query: str, file_ids: List[str], top_k: int = 5, client: MCPClient | None = None) -> str:
    """
    Compare/query across multiple PDFs. Returns combined context per file.
    """
    mcp = _get_client(client)
    result: List[str] = mcp.invoke(
        "compare_pdfs",
        {"query": query, "file_ids": file_ids, "top_k": top_k},
    )
    return "\n\n".join(result)


def tool_get_file_summaries(file_ids: List[str], client: MCPClient | None = None) -> str:
    """
    Lấy tóm tắt nội dung chính của các file. Dùng cho câu hỏi tổng quan hoặc so sánh bao quát.
    """
    if not file_ids:
        return "(Khong co file_id de tom tat.)"
    mcp = _get_client(client)
    results: List[str] = mcp.invoke("get_file_summaries", {"file_ids": file_ids})
    return "\n\n".join(results)
