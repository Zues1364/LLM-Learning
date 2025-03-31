from google import genai
from google.genai import types
import pathlib

# Khởi tạo client Gemini
client = genai.Client(api_key="AIzaSyDZHuPDlDang5rItvWKORmogW5AiDF5ga4")  # Thay bằng API key thực tế của bạn

# Đường dẫn đến tệp PDF cục bộ
local_pdf_path = "../data/pdfs/SỔ TAY HỌC VỤ KỲ I NĂM 2023-2024.pdf"  # Thay bằng đường dẫn thực tế đến tệp PDF của bạn

# Đọc tệp PDF từ đường dẫn cục bộ
filepath = pathlib.Path(local_pdf_path)

# Kiểm tra xem tệp có tồn tại không
if not filepath.exists():
    print(f"Tệp {local_pdf_path} không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    exit(1)

# Tạo prompt để yêu cầu tóm tắt
prompt = "Nói lại chi tiết về lịch ở trang 7 tài liệu này"

# Gửi yêu cầu đến Gemini API
response = client.models.generate_content(
    model="gemini-1.5-flash",  # Sử dụng model gemini-1.5-flash như trong ví dụ của bạn
    contents=[
        types.Part.from_bytes(
            data=filepath.read_bytes(),
            mime_type='application/pdf',
        ),
        prompt
    ]
)

# In kết quả tóm tắt
print(response.text)