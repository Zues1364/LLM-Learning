import os
import json
from typing import Any, Dict, List

from mcp_client.client import MCPClient


# Simple wrappers around MCP tools so we can register them as tools for an agent.
def _get_client(client: MCPClient | None) -> MCPClient:
    return client or MCPClient()


def tool_retrieve(question: str, top_k: int = 25, file_ids: List[str] | None = None,
                  session_id: str | None = None,
                  client: MCPClient | None = None) -> str:
    """
    Retrieve relevant chunks from the PDF store. Use when the question is about the uploaded PDF.
    Returns a concatenated text context.
    """
    mcp = _get_client(client)
    chunks: List[str] = mcp.invoke(
        "retrieve_chunks",
        {"question": question, "top_k": top_k, "file_ids": file_ids or [], "session_id": session_id},
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


def tool_compare_pdfs(query: str, file_ids: List[str], top_k: int = 25, client: MCPClient | None = None) -> str:
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

def tool_analyze_transcript(file_ids: List[str] | str, client: MCPClient | None = None) -> str:
    """
    Trich xuat bang diem sinh vien thanh JSON cau truc.
    """
    mcp = _get_client(client)

    result: Any = mcp.invoke("analyze_transcript", {"file_ids": file_ids})
    
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def tool_math_eval(expression: str, client: MCPClient | None = None) -> str:
    """
    May tinh an toan danh gia bieu thuc.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke("math_eval", {"expression": expression})
    return str(result)


def tool_consult_advisor(
    query: str,
    file_ids: List[str] | None = None,
    session_id: str = "default",
    program_id: str | None = None,
    client: MCPClient | None = None,
) -> str:
    """
    Goi Academic Advisor Agent qua MCP server.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "consult_advisor",
        {
            "query": query,
            "file_ids": file_ids or [],
            "session_id": session_id,
            "program_id": program_id,
        },
    )
    return str(result)


def tool_get_schedule(subject_codes: List[str], session_id: str | None = None, client: MCPClient | None = None) -> str:
    """
    Tim kiem lich hoc (TKB) cho danh sach cac ma mon hoc.
    Tra ve ket qua la cac dong chua ma mon hoc tu file TKB PDF toan truong.
    Input example: ["INT3306", "PEC1008"]
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke("get_schedule", {"subject_codes": subject_codes, "session_id": session_id})
    # JSON string is returned from server, so we just pass it through/ensure string
    return str(result)


def tool_get_available_programs(refresh: bool = False, client: MCPClient | None = None) -> str:
    """
    Lay danh sach cac chuong trinh dao tao co san trong he thong.
    He thong tu dong quet va nhan dien tu noi dung file HTML.
    
    Args:
        refresh: True de quet lai thu muc, False de dung cache.
    Returns:
        JSON danh sach [{id, name, year, display_name}]
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke("get_available_programs", {"refresh": refresh})
    return str(result)


def tool_get_curriculum_lookup(group_hint: str = None, program_id: str = None, session_id: str | None = None, client: MCPClient | None = None) -> str:
    """
    Tra cuu danh sach cac mon hoc trong chuong trinh dao tao.
    Dung khi nguoi dung hoi ve: hoc phan tu chon, mon nao con thieu, yeu cau tot nghiep, danh sach mon hoc trong CTDT.
    
    Args:
        group_hint: Vi du 'V.2.1', 'tu chon', 'Phan mem' de loc nhom mon. 
                    Neu None, tra ve tat ca nhom mon.
        program_id: Ma chuong trinh dao tao (vd: 'it_2025', 'cs_2022'). Neu None, dung chuong trinh mac dinh.
    Returns:
        JSON chua danh sach nhom mon va cac mon hoc tuong ung.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke("get_curriculum_lookup", {"group_hint": group_hint, "program_id": program_id, "session_id": session_id})
    return str(result)


def tool_get_electives_with_schedule(check_schedule: bool = True, program_id: str = None, session_id: str | None = None, client: MCPClient | None = None) -> str:
    """
    Lấy danh sách các môn TỰ CHỌN từ Chương trình Đào tạo VÀ kiểm tra xem môn nào đang MỞ trong TKB.
    Dùng khi nguoi dung hoi ve: 'hoc phan tu chon nao dang mo', 'mon tu chon trong ky nay', 
    'lua chon nao dang co lop', 'dang ky mon tu chon'.
    
    Args:
        check_schedule: True để kiểm tra TKB, False chỉ lấy danh sách từ CTĐT.
        program_id: Ma chuong trinh dao tao (vd: 'it_2025', 'cs_2022'). Neu None, dung chuong trinh mac dinh.
    Returns:
        JSON với "opened" (môn đang mở lớp) và "not_opened" (môn chưa mở)
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "get_electives_with_schedule",
        {"check_schedule": check_schedule, "program_id": program_id, "session_id": session_id},
    )
    return str(result)


def tool_resolve_course_alias(
    query: str,
    program_id: str | None = None,
    session_id: str | None = None,
    client: MCPClient | None = None,
) -> str:
    """
    Resolve ten mon/ma mon ve ma mon chuan trong du lieu TKB structured.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "resolve_course_alias",
        {"query": query, "program_id": program_id, "session_id": session_id},
    )
    return str(result)


def tool_get_teachers_by_subject(
    subject_code: str,
    semester: str | None = None,
    session_id: str | None = None,
    client: MCPClient | None = None,
) -> str:
    """
    Lay danh sach giang vien day mot mon hoc tu TKB structured.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "get_teachers_by_subject",
        {"subject_code": subject_code, "semester": semester, "session_id": session_id},
    )
    return str(result)


def tool_get_classes_by_teacher(
    teacher_name: str,
    semester: str | None = None,
    session_id: str | None = None,
    client: MCPClient | None = None,
) -> str:
    """
    Lay danh sach lop/mon theo giang vien tu TKB structured.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "get_classes_by_teacher",
        {"teacher_name": teacher_name, "semester": semester, "session_id": session_id},
    )
    return str(result)


def tool_get_schedule_rows(
    subject_code: str | None = None,
    teacher_name: str | None = None,
    semester: str | None = None,
    session_id: str | None = None,
    client: MCPClient | None = None,
) -> str:
    """
    Tra cuu dong lich hoc structured theo mon/giang vien.
    """
    mcp = _get_client(client)
    result: Any = mcp.invoke(
        "get_schedule_rows",
        {
            "subject_code": subject_code,
            "teacher_name": teacher_name,
            "semester": semester,
            "session_id": session_id,
        },
    )
    return str(result)
