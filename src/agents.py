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

from mcp_tools import (
    tool_retrieve,
    tool_web_search,
    tool_memory_get,
    tool_memory_add,
    tool_compare_pdfs,
    tool_get_file_summaries,
    tool_analyze_transcript,
    tool_consult_advisor,
    tool_math_eval,
    tool_get_schedule,
    tool_get_curriculum_lookup,
    tool_get_electives_with_schedule,
    tool_get_available_programs,
)


def _gemini_model(model_id: str) -> Gemini:
    """
    Build Gemini model with explicit GEMINI_API_KEY to avoid agno fallback warnings
    that hard-check GOOGLE_API_KEY.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set; Gemini model init may fail.")
        return Gemini(id=model_id)
    return Gemini(id=model_id, api_key=api_key)




# Planner Agent (tool-calling, tra JSON)

def get_mcp_planner_agent(allow_web_search: bool = False) -> Agent:
    """
    Tra ve JSON:
    {"source": "summary_index|vector_store|vector_store_compare|web_search|academic_advisor|error", "context": "...", "memory": "...", "chunk_index": int|null}
    """
    tools = [tool_get_file_summaries, tool_retrieve, tool_compare_pdfs, tool_memory_get, tool_consult_advisor, tool_get_curriculum_lookup, tool_get_electives_with_schedule, tool_get_available_programs]
    web_msg = "Ban khong duoc dung tool_web_search."
    if allow_web_search:
        tools.insert(2, tool_web_search)
        web_msg = "Neu tool_retrieve khong du, ban co the dung tool_web_search."

    instructions = (
        "Ban la planner. Dau vao luon co [SESSION:<id>] va co the co [FILES:f1,f2,...], [PROGRAM:<program_id>]. "
        "Luon truyen session_id tu [SESSION] vao tool_memory_get de lay lich su (ke ca khi rong). "
        
        "### PHASE 0 - PROGRAM IDENTIFICATION (CHUONG TRINH DAO TAO) ###: "
        "Uu tien doc [PROGRAM:<program_id>] trong CHINH dau vao. Neu da co [PROGRAM], BO QUA PHASE 0 va dung program_id nay cho moi tool hoc vu. "
        "Neu dau vao KHONG co [PROGRAM], kiem tra trong memory_context co [PROGRAM:<program_id>] chua. "
        "Neu KHONG co, goi tool tool_get_available_programs() de lay danh sach chuong trinh. "
        "Sau do tra JSON: {\"source\": \"program_selection\", \"context\": \"<danh sach chuong trinh tu tool>\", \"memory\": \"...\", \"chunk_index\": null, \"requires_selection\": true}. "
        "Front-end se hoi nguoi dung chon chuong trinh. Sau khi nguoi dung chon, session se co [PROGRAM:<id>]. "
        "Neu DA CO [PROGRAM:<program_id>], su dung program_id nay khi goi tool_get_curriculum_lookup va tool_get_electives_with_schedule. "
        
        "### UU TIEN 1 - HOC PHAN TU CHON MO LOP (ELECTIVES IN SCHEDULE) ###: "
        "Neu nguoi dung hoi bat ky cau hoi nao ve: 'mon tu chon', 'hoc phan tu chon', 'tu chon mo lop', 'tu chon ky nay', 'dang ky mon tu chon', 'mon nao dang mo', 'co lop nao mo', 'kiem tra mon tu chon', 'mon tu chon nao co lop': "
        "BAT BUOC goi tool tool_get_electives_with_schedule(check_schedule=True, program_id=<program_id neu co>). "
        "KHONG goi tool_retrieve hay tool_compare_pdfs cho cac cau hoi ve tu chon. "
        "Tra JSON voi source=electives_schedule, context=<ket qua tu tool>, memory=<ket qua tool_memory_get>, chunk_index=null. "
        
        "FOLLOW-UP SCHEDULING (UU TIEN CAO): Neu memory_context CHUA thong tin ve 'lich hoc', 'lap lich', 'TKB', 'Thứ 2', 'Thứ 3', 'tiết', hoặc 'học kỳ' VA cau hoi hien tai la follow-up nhu 'co', 'ok', 'them', 'bang lich', 'hoan chinh', 'them HIS1001', 'lich chi tiet': TUYET DOI HAY goi tool_consult_advisor(query, file_ids, session_id, program_id=<program_id neu co>). KHONG goi tool_compare_pdfs cho cac cau follow-up nhu vay. "
        "DYNAMIC ADVISORY: Neu nguoi dung hoi ve tinh diem, muc tieu GPA, hoac lo trinh hoc: TUYET DOI KHONG tu tra loi, KHONG goi tool_retrieve. Hay goi tool_consult_advisor(query, file_ids, session_id=[SESSION], program_id=<program_id neu co>). Ket qua tra ve tu tool nay chinh la context. Tra JSON voi source=academic_advisor, context=<ket qua tu tool>, memory=<ket qua tool_memory_get>, chunk_index=null. "

        "HOC PHAN TU CHON (ELECTIVES GENERAL - danh sach CTDT): Neu nguoi dung hoi ve 'danh sach hoc phan tu chon trong CTDT', 'cac mon tu chon la gi', 'danh sach mon trong chuong trinh': goi tool tool_get_curriculum_lookup(group_hint='tu chon', program_id=<program_id neu co>). Khong goi cai nay neu nguoi dung hoi ve 'mo lop', 'dang ky'. Tra JSON voi source=curriculum_lookup, context=<ket qua tu tool>, memory=<ket qua tool_memory_get>, chunk_index=null. "

        "Neu cau hoi mang tinh tong quan/khai quat/tom tat/noi dung chinh la gì hoac so sanh noi dung chinh giua cac file, va [FILES] co file_id, goi tool tool_get_file_summaries(file_ids) va dat source=summary_index (chunk_index=null). Neu khong co file_id, khong goi tool_get_file_summaries, dat source=error va context=Vui long cung cap file_ids. "
        "Neu hoi chi tiet ve >=2 file, goi tool tool_compare_pdfs(query, file_ids, top_k=15) va dat source=vector_store_compare (chi thuc hien khi co >=2 file_id). "
        "Neu hoi chi tiet mot file hoac [FILES] chi co 1 id, goi tool tool_retrieve(question, top_k=15, file_ids=[...]) va dat source=vector_store (chi thuc hien khi co file_id). "
        "TRUONG HOP DAC BIET: Neu [FILES] la rong (khong co file_id), ma nguoi dung hoi ve thong tin chung hoac quy che/so tay/chung chi ngoai ngu/IELTS/toeic/tot nghiep, hay goi `tool_retrieve(question, top_k=15, file_ids=[])`. He thong se tu dong tim trong cac Tai nguyen Toan cuc (Global Resources) nhu So tay, Quy che. Dat source=vector_store. "
        "QUAN TRONG - CHINH SACH/POLICY OVERRIDE: Neu cau hoi ve khainiem/dinhnghia/muctieu/hocphan/mien giam/chung chi: Bat buoc them tien to 'trong So tay hoc vu' vao question VA truyen `file_ids=[]` (rong) ngay ca khi co [FILES]. Dieu nay dam bao tim kiem toan cuc. "
        f"{web_msg} "
        "Luon truyen tham so file_ids khi goi tool_retrieve/tool_compare_pdfs/tool_get_file_summaries (lay tu [FILES], neu khong co thi truyen list rong []). "
        "Tra ve duy nhat MOT object JSON (khong phai list, khong code block) voi cac keys: source, context, memory, chunk_index. "
        "Vi du: {\"source\": \"curriculum_lookup\", \"context\": \"...\", \"memory\": \"...\", \"chunk_index\": null} "
        "LUU Y QUAN TRONG: Output phai la RAW JSON, khong duoc boc trong markdown ```json. Dam bao escape dau ngoac kep \" trong context neu co."
        "TUYET DOI KHONG TRA VE CHUOI RONG. Neu khong xac dinh duoc huong, tra JSON loi: "
        "{\"source\":\"error\",\"context\":\"Planner khong xac dinh duoc tool phu hop.\",\"memory\":\"\",\"chunk_index\":null}. "
        "Neu chunk_index la null, hay de la null (khong phai string 'null'). "
        "Neu loi tool, dat source=error va context la thong bao loi."
    )
    return Agent(
        name="MCP Planner Agent",
        model=_gemini_model("gemini-2.5-pro"),
        tools=tools,
        instructions=instructions,
        markdown=False,
    )


def get_academic_advisor_agent() -> Agent:
    instructions = """
    Vai tro: Ban la Academic Advisor formatter.
    Du lieu tinh toan deterministic da duoc cung cap trong CONTEXT.

    Nguyen tac boundary (bat buoc):
    - KHONG goi lai cac data tools de phan tich transcript/curriculum/schedule.
    - Khong lap lai pipeline; chi tong hop, giai thich va sap xep thong tin tu CONTEXT.

    Neu user hoi nhieu y trong 1 cau:
    - Tra loi trong MOT output duy nhat, khong yeu cau hoi tiep.
    - Luon trinh bay dung 4 muc:
      1) Thieu tin chi
      2) Mon con thieu uu tien
      3) GPA projection
      4) Goi y lich

    Quy tac trinh bay:
    - Uu tien so lieu cu the (tin chi, ma mon, GPA hien tai, max possible GPA, feasibility).
    - Trong muc "Thieu tin chi", bat buoc hien thi 3 so:
      1) Tin chi tich luy tren bang diem (transcript_total_credits)
      2) Tin chi duoc cong nhan theo CTDT (curriculum_applicable_credits)
      3) Tin chi con thieu (total_missing_credits)
    - Neu (1) khac (2), phai neu ro ly do (vi du: mon ngoai danh muc CTDT, hoac da duoc cong nhan theo nhom mo).
    - Neu co external_credits_applied, liet ke ma mon + so tin chi duoc cong nhan.
    - Neu du lieu lich hoc co "offered=False" hoac khong tim thay, noi ro mon chua mo.
    - Neu co canh bao mismatch ma mon, giu canh bao trong phan Goi y lich.
    - Ma mon co hau to "E" la ma hoc phan rieng, KHONG coi tuong duong voi ma khong "E".
    - Neu user yeu cau mon "...E" nhung TKB chi co mon khong "E": bat buoc ket luan mon "...E" chua mo, KHONG de xuat thay the mem.
    - TUYET DOI KHONG dung cum "thoi gian uoc tinh".
    - Chi duoc neu gio hoc khi co du lieu trong "resolved_time_range" hoac "time_slot_definitions".
    - Neu khong co gio hoc chinh xac, bat buoc ghi "chua xac dinh tu TKB nguon", khong duoc tu suy luan.
    - Neu context co "time_source_file", neu ro nguon nay khi trinh bay lich.
    - Khi trinh bay lich tuan dang bang, BAT BUOC dung 7 cot theo dung thu tu:
      Ngay hoc | Ca hoc | Tiet + Thoi gian | Ma mon hoc | Ten mon hoc | Tin chi | Ghi chu ve lop
    - TUYET DOI KHONG gop "Ca hoc" chung voi "Tiet + Thoi gian" trong mot cot.
    - Uu tien dung du lieu co san tu "schedule_table_rows" va "schedule_table_columns" trong CONTEXT.
    - Neu thieu mot phan du lieu nao trong CONTEXT, neu ro pham vi thieu nhung van tra loi phan con lai.

    Output:
    - Tieng Viet, gon, ro, co cau truc.
    """
    # Prevent tool_math_eval spam: cache by expression
    eval_cache: dict[str, str] = {}

    def safe_math_eval(expression: str) -> str:
        if expression in eval_cache:
            return eval_cache[expression]
        result = tool_math_eval(expression)
        eval_cache[expression] = result
        return result

    # Preserve tool name for clarity
    safe_math_eval.__name__ = "tool_math_eval"

    return Agent(
        name="Academic Advisor Agent",
        model=_gemini_model("gemini-2.5-flash"),
        tools=[safe_math_eval],
        instructions=instructions,
        markdown=True,
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
                f"Nguồn tham khảo: {source}\n\n"
                f"Câu hỏi: {query}\n\n"
                "Chỉ sử dụng thông tin từ Bối cảnh để trả lời chính. Lịch sử chỉ để tham chiếu ngữ cảnh hội thoại, không được ghi đè thông tin mới trong Bối cảnh. Nếu Bối cảnh có thông tin thì trả lời theo Bối cảnh. Nếu Bối cảnh trống, mới dùng thông tin từ Lịch sử. Trả lời ngắn gọn, tiếng Việt. TUYỆT ĐỐI KHÔNG ghi lại dòng 'Nguồn: ...' hoặc 'Nguồn tham khảo: ...' ở cuối câu trả lời."
            )
            logger.info("[AnswerGeneratorAgent] Action: Gọi LLM để sinh câu trả lời...")
            response = self.llm_agent.run(full_prompt)
            answer = response.content
            logger.info("[AnswerGeneratorAgent] Evaluation: Câu trả lời đã được sinh ra.")
            return answer
        except Exception as e:
            logger.error(f"[AnswerGeneratorAgent] Lỗi khi sinh câu trả lời: {e}")
            return f"Lỗi khi sinh câu trả lời: {e}"
# Gemini LLM Agent
def get_rag_agent() -> Agent:
    try:
        agent = Agent(
            name="Gemini RAG Agent",
            model=_gemini_model("gemini-2.5-flash"),
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


# Agent Scheduler
class SchedulerAgent:
    def __init__(self, llm_agent: Agent = None):
        try:
            self.llm_agent = llm_agent or get_scheduler_agent_internal()
        except NameError:
             # If internal factory is not defined yet (python linear execution), define it or lazy load
             # But here we define it below, so it's fine if class method calls it at runtime
             pass

    def run(self, context: str, subject_code: str) -> dict:
        try:
            # Lazy load if needed
            if not getattr(self, 'llm_agent', None):
                 self.llm_agent = get_scheduler_agent_internal()
                 
            logger.info(f"[SchedulerAgent] Observation: Extract schedule for {subject_code}")
            prompt = f"Trích xuất lịch học cho môn {subject_code} từ thông tin sau:\n\n{context}"
            response = self.llm_agent.run(prompt)
            raw_content = response.content
            
            # Clean possible markdown format
            if raw_content.startswith("```json"):
                raw_content = raw_content.replace("```json", "").replace("```", "").strip()
            
            # Try to parse
            data = json.loads(raw_content)
            logger.info(f"[SchedulerAgent] Extracted data for {subject_code}")
            return data
        except Exception as e:
            logger.error(f"[SchedulerAgent] Lỗi khi trích xuất lịch: {e}")
            return {}

def get_scheduler_agent_internal() -> Agent:
    instructions = (
        "Bạn là trợ lý lập lịch cá nhân. Nhiệm vụ của bạn là trích xuất thông tin lịch học từ văn bản được cung cấp.\n"
        "Đầu vào: Một đoạn văn bản chứa thông tin thời khóa biểu (có thể là bảng hoặc danh sách).\n"
        "Đầu ra: Trả về kết quả dưới dạng JSON với cấu trúc sau (nếu không tìm thấy thì để null/empty):\n"
        "[\n"
        "  {\n"
        "    \"subject_code\": \"Mã môn (ví dụ: INT3306)\",\n"
        "    \"subject_name\": \"Tên môn\",\n"
        "    \"class_code\": \"Mã lớp (ví dụ: INT3306 1)\",\n"
        "    \"credits\": Số tín chỉ,\n"
        "    \"schedule\": [\n"
        "       { \"day\": \"Thứ mấy (2,3,4,5,6,7,CN)\", \"period\": \"Tiết (ví dụ: 1-3)\", \"room\": \"Phòng học\" }\n"
        "    ],\n"
        "    \"lecturer\": \"Giảng viên (nếu có)\",\n"
        "    \"group\": \"Nhóm (CL, 1, 2...)\"\n"
        "  }\n"
        "]\n"
        "Chỉ trả về JSON thuần túy, không có Markdown block (```json)."
    )
    return Agent(
        name="Scheduler Internal Agent",
        model=_gemini_model("gemini-2.5-flash"),
        instructions=instructions,
        markdown=False,
    )

def get_scheduler_agent() -> SchedulerAgent:
    return SchedulerAgent()



