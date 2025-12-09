import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import json
from typing import List, Optional
from langchain_core.documents import Document
from agno.agent import Agent
from agno.models.google import Gemini
import requests
from mcp_client.client import MCPClient
from persistent_memory import PersistentMemory
from utils import FAISSVectorStore, web_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from mcp_tools import tool_retrieve, tool_web_search, tool_memory_get, tool_memory_add, tool_compare_pdfs, tool_get_file_summaries  




# Planner Agent (tool-calling, tra JSON)

def get_mcp_planner_agent(allow_web_search: bool = False) -> Agent:
    """
    Agent dang tool MCP de lay context/history. Tra ve JSON:
    {"source": "summary_index|vector_store|vector_store_compare|web_search|error", "context": "...", "memory": "...", "chunk_index": int|null}
    """
    tools = [tool_get_file_summaries, tool_retrieve, tool_compare_pdfs, tool_memory_get]
    web_msg = "Ban khong duoc dung web_search_tool."
    if allow_web_search:
        tools.insert(2, tool_web_search)
        web_msg = "Neu retrieve khong du, ban co the dung web_search_tool."

    instructions = (
        "Ban la planner. Dau vao luon co [SESSION:<id>] va co the co [FILES:f1,f2,...]. "
        "Luon truyen session_id tu [SESSION] vao memory_get de lay lich su (ke ca khi rong). "
        "Neu cau hoi mang tinh tong quan/khai quat/tom tat/“noi dung chinh la gi” hoac so sanh noi dung chinh giua cac file, va [FILES] co file_id, goi tool get_file_summaries(file_ids) va dat source=summary_index (chunk_index=null). Neu khong co file_id, khong goi get_file_summaries, dat source=error va context=Vui long cung cap file_ids. "
        "Neu hoi chi tiet ve >=2 file, goi tool compare_pdfs(query, file_ids, top_k=5) va dat source=vector_store_compare (chi thuc hien khi co >=2 file_id). "
        "Neu hoi chi tiet mot file hoac [FILES] chi co 1 id, goi tool retrieve(question, top_k=5, file_ids=[...]) va dat source=vector_store (chi thuc hien khi co file_id). "
        f"{web_msg} "
        "Luon truyen tham so file_ids khi goi retrieve/compare/get_file_summaries (lay tu [FILES], de rong neu khong co de fallback web). "
        "Tra ve duy nhat JSON (khong code block) voi khoa source, context, memory, chunk_index (co the null). "
        "Neu loi tool, dat source=error va context la thong bao loi."
    )
    return Agent(
        name="MCP Planner Agent",
        model=Gemini(id="gemini-2.5-flash"),
        tools=tools,
        instructions=instructions,
        markdown=False,
    )

# Agent Retriever
class RetrieverAgent:
    def __init__(self, vector_store: FAISSVectorStore, llm_agent: Agent = None):
        self.vector_store = vector_store
        self.llm_agent = llm_agent or get_ollama_agent()

    def run(self, query: str) -> tuple[str, str, Optional[int]]:
        try:
            logger.info(f"[RetrieverAgent] Observation: Nhận câu hỏi: {query}")
            optimized_query = query
            if self.llm_agent:
                logger.info("[RetrieverAgent] Thought: Dùng LLM để diễn giải câu hỏi...")
                prompt = f"Hãy diễn giải ngắn gọn (tiếng Việt) câu hỏi sau để tối ưu tìm kiếm tài liệu: {query}"
                response = self.llm_agent.run(prompt)
                candidate = (response.content or "").strip()
                logger.info(f"[RetrieverAgent] [DEBUG] LLM response candidate: {candidate[:200] if candidate else 'None'}...")
                if candidate and "Lỗi" not in candidate and len(candidate) <= 200:
                    optimized_query = candidate
                    logger.info(f"[RetrieverAgent] Câu hỏi đã được diễn giải: {optimized_query}")
                else:
                    logger.warning("[RetrieverAgent] Diễn giải không hợp lệ hoặc quá dài, dùng câu hỏi gốc.")

            logger.info(
                "[RetrieverAgent] Thought: Kiểm tra xem câu hỏi có thể được trả lời bằng tài liệu PDF hay không...")
            logger.info(f"[RetrieverAgent] [DEBUG] Query sử dụng cho retrieval: '{optimized_query}'")
            
            retrieved_docs = self.vector_store.retrieve(optimized_query, top_k=5)
            logger.info(f"[RetrieverAgent] [DEBUG] Số documents được retrieve: {len(retrieved_docs)}")
            
            chunk_index = None
            if not retrieved_docs:
                logger.info("[RetrieverAgent] [DEBUG] retrieved_docs is empty, fallback to web search")
                logger.info("[RetrieverAgent] Action: Không tìm thấy tài liệu, định tuyến đến Web Search...")
                return "web_search", "", chunk_index
            
            logger.info(f"[RetrieverAgent] [DEBUG] Tìm thấy {len(retrieved_docs)} documents, chuẩn bị trả về context")
            logger.info("[RetrieverAgent] Action: Truy xuất tài liệu từ Vector Store...")
            context = "\n\n".join([f"Chunk {doc.metadata.get('index')}: {doc.page_content}" for doc in retrieved_docs])
            chunk_index = retrieved_docs[0].metadata.get('index')
            logger.info(f"[RetrieverAgent] [DEBUG] Context length: {len(context)} characters")
            logger.info(f"[RetrieverAgent] [DEBUG] First chunk index: {chunk_index}")
            logger.info("[RetrieverAgent] Evaluation: Tài liệu truy xuất thành công.")
            return "vector_store", context, chunk_index
        except Exception as e:
            logger.error(f"[RetrieverAgent] Lỗi khi xử lý: {e}")
            return "error", f"Lỗi khi truy xuất tài liệu: {e}", None


# Agent Web Searcher
class WebSearcherAgent:
    def __init__(self, llm_agent: Agent = None):
        self.llm_agent = llm_agent or get_ollama_agent()

    def run(self, query: str) -> str:
        try:
            logger.info(f"[WebSearcherAgent] Observation: Nhận yêu cầu tìm kiếm web cho câu hỏi: {query}")
            logger.info("[WebSearcherAgent] Thought: Tạo truy vấn tìm kiếm web...")
            logger.info("[WebSearcherAgent] Action: Gọi API tìm kiếm web...")
            results = web_search(query)
            context = "\n".join(results)
            if self.llm_agent:
                logger.info("[WebSearcherAgent] Thought: Dùng LLM để tóm tắt kết quả web...")
                prompt = (
                    f"Dưới đây là kết quả tìm kiếm web cho câu hỏi: {query}\n\n"
                    f"Kết quả: {context}\n\n"
                    "Hãy tóm tắt kết quả trên thành một đoạn văn ngắn, chỉ giữ lại thông tin quan trọng và loại bỏ phần không liên quan."
                )
                response = self.llm_agent.run(prompt)
                summarized_context = response.content
                logger.info("[WebSearcherAgent] Kết quả đã được tóm tắt.")
            else:
                summarized_context = context

            logger.info("[WebSearcherAgent] Evaluation: Kết quả web đã được xử lý.")
            return summarized_context
        except Exception as e:
            logger.error(f"[WebSearcherAgent] Lỗi khi tìm kiếm web: {e}")
            return f"Lỗi khi tìm kiếm web: {e}"


# Agent Memory Manager
class MemoryManagerAgent:
    def __init__(self, memory: PersistentMemory, llm_agent: Agent = None):
        self.memory = memory
        self.llm_agent = llm_agent or get_ollama_agent()

    def run(self, query: str, session_id: str, chunk_index: Optional[int]) -> str:
        try:
            logger.info(f"[MemoryManagerAgent] Observation: Nhận yêu cầu truy xuất lịch sử cho session: {session_id}")
            logger.info("[MemoryManagerAgent] Thought: Truy xuất lịch sử liên quan...")
            logger.info("[MemoryManagerAgent] Action: Gọi hàm get_context...")
            raw_context = self.memory.get_context(query, session_id, chunk_index, max_rows=5)
            if self.llm_agent and raw_context:
                logger.info("[MemoryManagerAgent] Thought: Dùng LLM để phân tích lịch sử...")
                prompt = (
                    f"Dưới đây là lịch sử trò chuyện:\n{raw_context}\n\n"
                    f"Câu hỏi hiện tại: {query}\n\n"
                    "Hãy tóm tắt lịch sử trò chuyện, chỉ giữ lại thông tin liên quan đến câu hỏi hiện tại."
                )
                response = self.llm_agent.run(prompt)
                summarized_context = response.content
                logger.info("[MemoryManagerAgent] Lịch sử đã được tóm tắt.")
            else:
                summarized_context = raw_context

            logger.info("[MemoryManagerAgent] Evaluation: Lịch sử đã được xử lý.")
            return summarized_context
        except Exception as e:
            logger.error(f"[MemoryManagerAgent] Lỗi khi truy xuất lịch sử: {e}")
            return f"Lỗi khi truy xuất lịch sử: {e}"


# Agent Answer Generator
class AnswerGeneratorAgent:
    def __init__(self, llm_agent: Agent):
        self.llm_agent = llm_agent

    def run(self, query: str, context: str, source: str, memory_context: str) -> str:
        try:
            logger.info(f"[AnswerGeneratorAgent] Observation: Nhận câu hỏi và ngữ cảnh: {query}")
            logger.info("[AnswerGeneratorAgent] Thought: Xây dựng prompt và tổ chức câu trả lời...")
            full_prompt = (
                f"Bối cảnh: {context}\n\n"
                f"Lịch sử trò chuyện: {memory_context}\n\n"
                f"Nguồn: {source}\n\n"
                f"Câu hỏi: {query}\n\n"
                "Chỉ sử dụng thông tin từ Bối cảnh để trả lời chính. Lịch sử chỉ để tham chiếu ngữ cảnh hội thoại, không được ghi đè thông tin mới trong Bối cảnh. Nếu Bối cảnh có thông tin thì trả lời theo Bối cảnh. Nếu Bối cảnh trống, mới dùng thông tin từ Lịch sử. Trả lời ngắn gọn, tiếng Việt."
            )
            logger.info("[AnswerGeneratorAgent] Action: Gọi LLM để sinh câu trả lời...")
            response = self.llm_agent.run(full_prompt)
            answer = response.content
            logger.info("[AnswerGeneratorAgent] Evaluation: Câu trả lời đã được sinh ra.")
            return answer
        except Exception as e:
            logger.error(f"[AnswerGeneratorAgent] Lỗi khi sinh câu trả lời: {e}")
            return f"Lỗi khi sinh câu trả lời: {e}"


# Agent Coordinator
class CoordinatorAgent:
    def __init__(self, retriever: RetrieverAgent, web_searcher: WebSearcherAgent, memory_manager: MemoryManagerAgent,
                 answer_generator: AnswerGeneratorAgent, memory: PersistentMemory):
        self.retriever = retriever
        self.web_searcher = web_searcher
        self.memory_manager = memory_manager
        self.answer_generator = answer_generator
        self.memory = memory

    def run(self, query: str, session_id: str) -> str:
        try:
            logger.info(f"[CoordinatorAgent] Observation: Nhận câu hỏi từ người dùng: {query}")
            logger.info("[CoordinatorAgent] Thought: Điều phối các agent để xử lý câu hỏi...")

            logger.info("[CoordinatorAgent] Action: Kích hoạt RetrieverAgent...")
            source, context, chunk_index = self.retriever.run(query)
            if source == "error":
                logger.error("[CoordinatorAgent] Evaluation: RetrieverAgent trả về lỗi.")
                return context

            if source == "web_search":
                logger.info("[CoordinatorAgent] Action: Kích hoạt WebSearcherAgent...")
                context = self.web_searcher.run(query)
                if context.startswith("Lỗi"):
                    logger.error("[CoordinatorAgent] Evaluation: WebSearcherAgent trả về lỗi.")
                    return context

            logger.info("[CoordinatorAgent] Action: Kích hoạt MemoryManagerAgent...")
            memory_context = self.memory_manager.run(query, session_id, chunk_index)
            if memory_context.startswith("Lỗi"):
                logger.error("[CoordinatorAgent] Evaluation: MemoryManagerAgent trả về lỗi.")
                return memory_context

            logger.info("[CoordinatorAgent] Action: Kích hoạt AnswerGeneratorAgent...")
            answer = self.answer_generator.run(query, context, source, memory_context)
            if answer.startswith("Lỗi"):
                logger.error("[CoordinatorAgent] Evaluation: AnswerGeneratorAgent trả về lỗi.")
                return answer

            logger.info("[CoordinatorAgent] Action: Lưu câu trả lời vào lịch sử...")
            self.memory.add_to_history(query, answer, session_id, chunk_index)

            logger.info("[CoordinatorAgent] Evaluation: Hoàn thành xử lý câu hỏi.")
            return answer
        except Exception as e:
            logger.error(f"[CoordinatorAgent] Lỗi khi điều phối: {e}")
            return f"Lỗi khi điều phối các agent: {e}"


# Gemini LLM Agent
def get_rag_agent() -> Agent:
    try:
        agent = Agent(
            name="Gemini RAG Agent",
            model=Gemini(id="gemini-2.5-flash"),
            instructions=(
                "Ban la tro ly RAG, tra loi chinh xac dua tren tai lieu.\n"
                "Neu thong tin den tu PDF, tra loi chi tiet va neu ro ten file/Chunk neu co metadata file_name.\n"
                "Neu thong tin den tu web, ghi ro nguon la 'Web Search'.\n"
                "Neu co lich su trao doi trong context, dung de boi canh nhung khong bia thong tin moi.\n"
                "Neu context khong du, noi ro rang thay vi doan.\n"
                "Tra loi tieng Viet, ngan gon, ro rang."
            ),
            markdown=True,
        )
        logger.info("[get_rag_agent] Da tao RAG Agent thanh cong.")
        return agent
    except Exception as e:
        logger.error(f"[get_rag_agent] Loi khi tao RAG Agent: {e}")
        raise

# Ollama Agent
def get_ollama_agent(model_name: str = "llama3") -> Agent:
    class OllamaAgent(Agent):
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.base_url = "http://localhost:11434"

        def run(self, prompt: str) -> type('Response', (), {'content': ''}):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return type('Response', (), {'content': result.get('response', 'Lỗi khi gọi Ollama')})()
            except Exception as e:
                logger.error(f"[OllamaAgent] Lỗi khi gọi Ollama: {e}")
                return type('Response', (), {'content': f"Lỗi khi gọi Ollama: {e}"})()

    try:
        agent = OllamaAgent(model_name)
        logger.info(f"[get_ollama_agent] Đã tạo Ollama Agent với mô hình {model_name} thành công.")
        return agent
    except Exception as e:
        logger.error(f"[get_ollama_agent] Lỗi khi tạo Ollama Agent: {e}")
        raise
