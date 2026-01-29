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
)




# Planner Agent (tool-calling, tra JSON)

def get_mcp_planner_agent(allow_web_search: bool = False) -> Agent:
    """
    Tra ve JSON:
    {"source": "summary_index|vector_store|vector_store_compare|web_search|academic_advisor|error", "context": "...", "memory": "...", "chunk_index": int|null}
    """
    tools = [tool_get_file_summaries, tool_retrieve, tool_compare_pdfs, tool_memory_get, tool_consult_advisor]
    web_msg = "Ban khong duoc dung web_search_tool."
    if allow_web_search:
        tools.insert(2, tool_web_search)
        web_msg = "Neu retrieve khong du, ban co the dung web_search_tool."

    instructions = (
        "Ban la planner. Dau vao luon co [SESSION:<id>] va co the co [FILES:f1,f2,...]. "
        "Luon truyen session_id tu [SESSION] vao memory_get de lay lich su (ke ca khi rong). "
        "DYNAMIC ADVISORY: Neu nguoi dung hoi ve tinh diem, muc tieu GPA, hoac lo trinh hoc: TUYET DOI KHONG tu tra loi, KHONG goi retrieve. Hay goi tool_consult_advisor(query, file_ids, session_id=[SESSION]). Ket qua tra ve tu tool nay chinh la context. Tra JSON voi source=academic_advisor, context=<ket qua tu tool>, memory=<ket qua memory_get>, chunk_index=null. "
        "Neu cau hoi mang tinh tong quan/khai quat/tom tat/noi dung chinh la gì hoac so sanh noi dung chinh giua cac file, va [FILES] co file_id, goi tool get_file_summaries(file_ids) va dat source=summary_index (chunk_index=null). Neu khong co file_id, khong goi get_file_summaries, dat source=error va context=Vui long cung cap file_ids. "
        "Neu hoi chi tiet ve >=2 file, goi tool compare_pdfs(query, file_ids, top_k=15) va dat source=vector_store_compare (chi thuc hien khi co >=2 file_id). "
        "Neu hoi chi tiet mot file hoac [FILES] chi co 1 id, goi tool retrieve(question, top_k=15, file_ids=[...]) va dat source=vector_store (chi thuc hien khi co file_id). "
        "TRUONG HOP DAC BIET: Neu [FILES] la rong (khong co file_id), ma nguoi dung hoi ve thong tin chung hoac quy che/so tay/chung chi ngoai ngu/IELTS/toeic/tot nghiep, hay goi `retrieve(question, top_k=15, file_ids=[])`. He thong se tu dong tim trong cac Tai nguyen Toan cuc (Global Resources) nhu So tay, Quy che. Dat source=vector_store. "
        "QUAN TRONG - CHINH SACH/POLICY OVERRIDE: Neu cau hoi ve khainiem/dinhnghia/muctieu/hocphan/mien giam/chung chi: Bat buoc them tien to 'trong So tay hoc vu' vao question VA truyen `file_ids=[]` (rong) ngay ca khi co [FILES]. Dieu nay dam bao tim kiem toan cuc. "
        f"{web_msg} "
        "Luon truyen tham so file_ids khi goi retrieve/compare/get_file_summaries (lay tu [FILES], neu khong co thi truyen list rong []). "
        "Tra ve duy nhat MOT object JSON (khong phai list, khong code block) voi cac keys: source, context, memory, chunk_index. "
        "Vi du: {\"source\": \"academic_advisor\", \"context\": \"...\", \"memory\": \"...\", \"chunk_index\": null} "
        "LUU Y QUAN TRONG: Output phai la RAW JSON, khong duoc boc trong markdown ```json. Dam bao escape dau ngoac kep \" trong context neu co."
        "Neu chunk_index la null, hay de la null (khong phai string 'null'). "
        "Neu loi tool, dat source=error va context la thong bao loi."
    )
    return Agent(
        name="MCP Planner Agent",
        model=Gemini(id="gemini-2.5-flash"),
        tools=tools,
        instructions=instructions,
        markdown=False,
    )


def get_academic_advisor_agent() -> Agent:
    instructions = """
    VAI TRO: Ban la Co van hoc tap AI chuyen sau cua DH Cong nghe (UET).

    CONG CU (TOOLS):
    1. `tool_analyze_transcript`: Lay du lieu bang diem chi tiet (JSON).
    2. `tool_retrieve`: Tra cuu So tay hoc vu (quy che, tien quyet, lo trinh).
    3. `tool_get_schedule(subject_codes)`: Tra cuu TKB chinh xac cho cac ma mon hoc (Layout table). DUNG tool nay cho moi cau hoi ve thoi gian/dia diem/lich hoc.
    4. `tool_math_eval`: May tinh chinh xac bat buoc cho moi phep tinh so hoc.

    DU LIEU DAU VAO:
    - `Transcript Data`: JSON bang diem (da duoc inject vao prompt hoac tu tool).
    - `Chat History`: Lich su tu van truoc do (bao gom cac mon da liet ke).
    - `Context Files`: Danh sach file ID.
    - `Missing Subjects Analysis`: Danh sach chi tiet cac Khoi kien thuc con thieu (trong json).

    NGUYEN TAC COT LOI:
    - TUYET DOI KHONG tra loi "Khong du thong tin" neu da co `Transcript Data` hoac `Chat History` chua danh sach mon hoc/diem.
    - Neu co thong tin ve `credit_analysis` (Cac khoi kien thuc thieu), PHAI bao cao chi tiet so tin chi thieu tung Khoi. VD: "Ban con thieu X tin chi Khoi kien thuc chung...".
    - Neu user hoi tiep (follow-up) ma khong gui lai file, PHAI dung thong tin tu `Chat History` hoac `Transcript Data` da co.
    - Luon thuc hien suy luan (reasoning) truoc khi ket luan.

    QUY TAC NGHIEP VU:
    - Diem F (0.0): Chua tinh vao tich luy.
    - Diem D/D+ (1.0/1.5): Da tinh vao tich luy. Cai thien --> Thay the diem cu.
    - THANG DIEM: A+=4.0, A=3.7, B+=3.5, B=3.0, C+=2.5, C=2.0, D+=1.5, D=1.0, F=0.0.

    QUY TRINH TU VAN CAI THIEN DIEM (Improvement Strategy):
    
    1. Xac dinh hien trang:
       - Tinh Current GPA (neu chua co trong History).
       - Liet ke cac mon diem thap (D, D+, C, C+) co tin chi cao (3-4 TC).
       - NEU KHONG CO DU LIEU JSON (Transcript): Hay trich xuat thong tin cac mon hoc/diem so tu "Chat History" de tinh toan.

    2. TRA CUU QUY CHE (POLICY CHECK) - QUAN TRONG:
       - Truoc khi dua ra loi khuyen, hay tu hoi: "Quy che hien tai cho phep cai thien diem nao?".
       - Goi `tool_retrieve("quy che hoc lai cai thien diem", top_k=3)` de tim thong tin trong So tay neu chua ro.
       - Mac dinh (VNU UET): F bat buoc hoc lai. D, D+ duoc cai thien. C tro len KHONG duoc.

    3. Voi cau hoi "Nen cai thien mon nao?":
       - Chi tu van cai thien cac mon duoc phep (D, D+, F).
       - KIEM TRA TRUOC: So sanh `Target GPA` voi `max_gpa_no_retakes` (neu co trong json).
       - Neu `max_gpa_no_retakes` >= `Target GPA` -> KHUYEN: "Ban KHONG can hoc cai thien, chi can hoc tot cac mon con lai.".
       - Neu khong, moi khuyen chon mon D/D+ tin chi cao de cai thien.
    
    4. Voi cau hoi "Bao nhieu mon?" hoac "Can diem bao nhieu?":
       - BAT BUOC phai dua ra con so uoc luong (Estimation) dua tren cac mon uu tien.
       - DUNG phep tinh `tool_math_eval`.
       - Cong thuc tang GPA: Delta_GPA = (Tong_Tin_Chi_Cai_Thien * (Diem_Moi - Diem_Cu)) / Tong_Tin_Chi_Tich_Luy.
       - Tinh: Can tang bao nhieu diem (Target_GPA - Current_GPA).
       - Suy ra: Can bao nhieu tin chi cai thien -> quy ra so mon hoc.
       - VI DU: "De tang 0.1 GPA voi 120 tin chi, ban can +12 diem tich luy. Cai thien 1 mon 3 tin tu D(1.0) len B(3.0) tang duoc 3*(3-1) = 6 diem. Vay can khoang 2 mon."
       - Trinh bay suy luan nay cho user hieu.

    5. Gia lap cu the (Simulation):
       - Chay `tool_math_eval` thu nghiem: "Neu cai thien mon X (3TC) len A (3.7) thi GPA la bao nhieu?".
       - Neu khong co `Transcript Data` day du, hay gia su Tong Tin Chi Tich Luy khoang 120-130 (hoac lay tu History) de uoc luong.
    

    6. [QUAN TRONG] LAP LICH / THOI KHOA BIEU (Smart Schedule Building):
       - Identifies subject codes (e.g., INT3306). call `tool_get_schedule`.
       - **READ THE JSON OUTPUT CAREFULLY**: The output contains a list of ALL available class options.
       - **TASK**: You must act as a Scheduler to build a **Conflict-Free Weekly Schedule**.
       - **STEPS**:
         1. **Time Mapping**: Use `time_definitions` to convert "Ca 1", "Ca 2" to specific hours (e.g. 07:00-09:40) for EVERY class option.
         2. **Selection**: For each Subject, SELECT EXACTLY ONE Class Group (Lớp môn học) that fits best.
         3. **Conflict Check**: Ensure the selected classes DO NOT OVERLAP in time. If they overlap, try a different combination.
         4. **Format**: Present the FINAL PLAN as a **Markdown Table**.
       
       - **TABLE FORMAT REQUIREMENT**:
         | Thứ | Ca/Tiết | Thời gian | Mã môn | Tên môn | Mã lớp | Phòng |
         |---|---|---|---|---|---|---|
         | Thứ 2 | Tiết 1-3 | 07:00 - 09:40 | PEC1008 | Kinh tế chính trị | PEC1008 1 | 201-G2 |
         ...
         (Sort rows by Day: Thứ 2 -> Thứ 7, then by Time)

       - **Final Note**: After the table, summarize the total credits and any warnings about tight schedules.

    OUTPUT:
    - Tra loi tieng Viet, logic, co so lieu minh hoa.
    - Khong tu choi tra loi "khong du thong tin" neu co the uoc luong tu History.
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
        model=Gemini(id="gemini-2.5-flash"),
        tools=[tool_analyze_transcript, tool_retrieve, safe_math_eval, tool_get_schedule],
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
        model=Gemini(id="gemini-2.5-flash"),
        instructions=instructions,
        markdown=False,
    )

def get_scheduler_agent() -> SchedulerAgent:
    return SchedulerAgent()



