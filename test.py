import os
from google import genai
from dotenv import load_dotenv

# 1. Tải API Key từ file .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "dien_api_key_cua_ban_vao_day":
    print("❌ LỖI: Bạn chưa điền GEMINI_API_KEY vào file .env")
    exit()

# 2. Khởi tạo Client Gemini
client = genai.Client(api_key=api_key)

print("⏳ Đang kết nối tới Google Gemini (phiên bản mới)...")
try:
    print("\n✅ HỆ THỐNG ĐÃ SẴN SÀNG!")
    print("Bạn có thể bắt đầu chat thử ngay dưới đây (Gõ 'thoát' để dừng).")
    print("-" * 60)
    
    # 3. Vòng lặp Chat trên Terminal
    try:
        while True:
            user_input = input("\nBạn: ")
            
            # Xử lý khi user chỉ nhấn Enter mà chưa gõ gì
            if not user_input.strip():
                continue
                
            if user_input.lower().strip() == 'thoát':
                print("Đã thoát ứng dụng test.")
                break
                
            print("Chatbot: ", end="", flush=True)
            
            # Gửi request và nhận stream
            response = client.models.generate_content_stream(
                model='gemini-3.5-flash', 
                contents=user_input
            )
            
            # Lặp qua từng khối dữ liệu trả về và in ra
            for chunk in response:
                if chunk.text:
                    import sys, time
                    for char in chunk.text:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                        time.sleep(0.01) # Tạo độ trễ nhỏ để nhìn rõ hiệu ứng stream
            print("\n") # Xuống dòng khi bot in xong
    except KeyboardInterrupt:
        print("\n\nĐã thoát ứng dụng bằng tổ hợp phím (Ctrl+C). Tạm biệt!")
        
except Exception as e:
    print(f"\n❌ LỖI KẾT NỐI: {e}")
    print("Vui lòng kiểm tra lại API Key hoặc mạng internet của bạn.")
