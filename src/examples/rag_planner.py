import os
import sys
import logging
import google.generativeai as genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import AnswerGeneratorAgent, get_rag_agent  
from mcp_client.client import MCPClient  
from env_loader import load_env  


load_env()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Add it to .env or environment variables.")
genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger(__name__)


class RAGPlanner:
    def __init__(self, mcp_server: str | None = None):
        self.mcp = MCPClient(mcp_server)
        self.answer_llm = AnswerGeneratorAgent(get_rag_agent())

    def run(self, query: str, session_id: str = "default") -> str:
        logger.info("Planner nhận câu hỏi: %s", query)

        # 1) Lấy lịch sử gần nhất
        history_lines = self.mcp.invoke(
            "memory_get", {"session_id": session_id, "max_rows": 5}
        )
        history_ctx = "\n".join(history_lines)

        # 2) Gọi retrieve_chunks trước
        chunks = self.mcp.invoke(
            "retrieve_chunks", {"question": query, "top_k": 3}
        )
        source = "vector_store" if chunks else "web_search"

        if not chunks:  # Fallback sang web search
            snippets = self.mcp.invoke(
                "web_search_tool", {"query": query, "num_results": 5}
            )
            chunks = snippets  # Đang snippets làm context

        context = "\n\n".join(chunks)

        # 3) Sinh câu trả lời
        answer = self.answer_llm.run(
            query=query,
            context=context,
            source=source,
            memory_context=history_ctx,
        )

        # 4) Lưu vào memory qua MCP
        self.mcp.invoke(
            "memory_add",
            {
                "session_id": session_id,
                "query": query,
                "answer": answer,
                "chunk_index": None,
            },
        )

        return answer
