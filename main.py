import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
from google import genai
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env
load_dotenv()

# Cấu hình Google Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key or api_key == "dien_api_key_cua_ban_vao_day":
    print("CẢNH BÁO: Bạn chưa thiết lập GEMINI_API_KEY trong file .env")

client = genai.Client(api_key=api_key)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Sales Assistant API")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080").rstrip("/")

# Cấu hình CORS để cho phép React Frontend kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn hoặc bạn có thể chỉ định ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo ChromaDB - Lưu trữ dữ liệu cục bộ trong thư mục 'chroma_data'
chroma_client = chromadb.PersistentClient(path="./chroma_data")
# Tạo hoặc lấy bảng (collection) lưu sản phẩm
collection = chroma_client.get_or_create_collection(name="products")

# Cấu trúc dữ liệu Biến thể của Sản phẩm
class VariantData(BaseModel):
    size: str | None = None
    color: str | None = None
    price: float | None = None
    salePrice: float | None = None
    stockQuantity: int = 0

# Cấu trúc dữ liệu Sản phẩm nhận từ Java
class ProductData(BaseModel):
    id: str
    name: str
    price: float
    basePrice: float | None = None
    discountPrice: float | None = None
    description: str
    category: str
    variants: list[VariantData] | None = None

# Cấu trúc dữ liệu Tin nhắn nhận từ Frontend
class ChatMessage(BaseModel):
    message: str

import requests

# Hàm kiểm tra sản phẩm có phải bán chạy nhất không
def check_is_best_seller(p_id: str) -> bool:
    try:
        bs_response = requests.get(f"{BACKEND_URL}/api/products/best-sellers")
        if bs_response.status_code == 200:
            bs_data = bs_response.json()
            for item in bs_data:
                if str(item.get("id")) == p_id:
                    return True
    except Exception as e:
        print("Lỗi kiểm tra best seller:", e)
    return False

# Hàm dựng tài liệu mô tả chi tiết sản phẩm cho RAG
def build_product_document(
    name: str, 
    p_id: str, 
    category: str, 
    base_price: float | None, 
    discount_price: float | None, 
    description: str, 
    variants: list | None, 
    is_best_seller: bool = False
) -> str:
    # 1. Khuyến mãi và giá cả
    price_info = ""
    if discount_price and base_price and discount_price < base_price:
        saved = base_price - discount_price
        price_info = f"Giá gốc: {base_price:,.0f} VNĐ, Giá khuyến mãi hiện tại: {discount_price:,.0f} VNĐ (Đang GIẢM GIÁ ưu đãi giảm {saved:,.0f} VNĐ!)."
    elif base_price:
        price_info = f"Giá bán: {base_price:,.0f} VNĐ."
    else:
        price_info = f"Giá bán: {discount_price:,.0f} VNĐ." if discount_price else "Liên hệ cửa hàng để biết giá cụ thể."

    # 2. Nhãn bán chạy
    best_seller_label = ""
    if is_best_seller:
        best_seller_label = "\nĐặc điểm nổi bật: Sản phẩm bán chạy nhất (Best Seller) của cửa hàng Chinh Hoops!"

    # 3. Thông tin biến thể (size, màu, tồn kho)
    variants_info = []
    if variants:
        for v in variants:
            if hasattr(v, "model_dump"):  # Pydantic model
                v_dict = v.model_dump()
            else:
                v_dict = v
            
            size = v_dict.get("size")
            color = v_dict.get("color")
            v_price = v_dict.get("salePrice") if v_dict.get("salePrice") else v_dict.get("price")
            # Nếu biến thể không có giá riêng, lấy giá của sản phẩm chính làm mặc định
            if not v_price:
                v_price = discount_price if discount_price else base_price
                
            stock = v_dict.get("stockQuantity", 0)
            
            v_price_str = ""
            if v_price:
                v_price_str = f" với giá {v_price:,.0f} VNĐ"
            
            stock_str = f"còn {stock} đôi" if stock > 0 else "đã HẾT HÀNG"
            variants_info.append(f"- Màu {color}, Kích cỡ {size}: {stock_str}{v_price_str}.")
    
    variants_text = "\n".join(variants_info) if variants_info else "Sản phẩm không có phân loại size/màu cụ thể."

    doc_text = f"""Tên sản phẩm: {name}
Mã sản phẩm (ID): {p_id}
Danh mục sản phẩm: {category}
Thông tin giá bán: {price_info}{best_seller_label}
Mô tả sản phẩm: {description}
Thông tin chi tiết các biến thể (Màu sắc, Kích cỡ & Tồn kho):
{variants_text}
"""
    return doc_text

@app.post("/api/ai/sync-all-from-source")
async def sync_all_from_source():
    """API đặc biệt: Gọi 1 lần để kéo toàn bộ dữ liệu từ Java sang AI (Pull)"""
    try:
        # Gọi API của Java để lấy danh sách sản phẩm
        java_api_url = f"{BACKEND_URL}/api/products?size=10000"
        response = requests.get(java_api_url)
        if response.status_code != 200:
            return {"status": "error", "message": "Không thể kết nối đến Java Server"}
        
        data = response.json()
        products = data.get("content", [])
        
        # Lấy danh sách sản phẩm bán chạy để đánh nhãn
        best_sellers_ids = []
        try:
            bs_response = requests.get(f"{BACKEND_URL}/api/products/best-sellers")
            if bs_response.status_code == 200:
                bs_data = bs_response.json()
                best_sellers_ids = [str(item.get("id")) for item in bs_data]
        except Exception as e:
            print("Lỗi lấy danh sách best sellers:", e)
        
        count = 0
        for p in products:
            p_id = str(p.get("id"))
            desc = p.get("shortDescription", "")
            if p.get("description"):
                desc += " - " + p.get("description")
                
            is_best_seller = p_id in best_sellers_ids
            
            # Xây dựng document văn bản chi tiết
            document_text = build_product_document(
                name=p.get("name", ""),
                p_id=p_id,
                category=p.get("categoryName", ""),
                base_price=p.get("basePrice"),
                discount_price=p.get("discountPrice"),
                description=desc,
                variants=p.get("variants"),
                is_best_seller=is_best_seller
            )
            
            collection.upsert(
                documents=[document_text],
                metadatas=[{"id": p_id, "name": p.get("name", ""), "price": float(p.get("price", 0))}],
                ids=[p_id]
            )
            count += 1
            
        return {"status": "success", "message": f"Đã học (Sync) thành công {count} sản phẩm từ Java (kèm các biến thể chi tiết và khuyến mãi)."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/sync")
async def sync_product(product: ProductData):
    """API để Java gọi sang mỗi khi Thêm/Sửa sản phẩm"""
    try:
        # Kiểm tra trạng thái bán chạy
        is_best_seller = check_is_best_seller(product.id)
        
        # Xây dựng document văn bản chi tiết
        document_text = build_product_document(
            name=product.name,
            p_id=product.id,
            category=product.category,
            base_price=product.basePrice,
            discount_price=product.discountPrice,
            description=product.description,
            variants=product.variants,
            is_best_seller=is_best_seller
        )
        
        collection.upsert(
            documents=[document_text],
            metadatas=[{"id": product.id, "name": product.name, "price": product.price}],
            ids=[str(product.id)]
        )
        return {"status": "success", "message": f"Đã đồng bộ sản phẩm kèm biến thể: {product.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/sync/delete")
async def delete_product(product_id: str):
    """API để Java gọi sang mỗi khi Xóa sản phẩm"""
    try:
        collection.delete(ids=[product_id])
        return {"status": "success", "message": f"Đã xóa sản phẩm ID: {product_id} khỏi AI"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/chat")
async def chat_bot(chat: ChatMessage):
    """API để Frontend gọi vào khi Khách nhắn tin"""
    try:
        import asyncio
        
        # Kiểm tra xem câu hỏi có phải dạng tổng quan hoặc phân tích toàn cửa hàng không
        msg_lower = chat.message.lower()
        global_keywords = ["đắt nhất", "mắc nhất", "cao nhất", "rẻ nhất", "thấp nhất", "bán chạy", "best seller", "hot nhất", "tất cả", "danh sách", "bao nhiêu sản phẩm", "phổ biến nhất"]
        is_global = any(kw in msg_lower for kw in global_keywords)
        
        if is_global:
            # Lấy toàn bộ sản phẩm trong ChromaDB để AI có cái nhìn toàn cảnh
            results = await asyncio.to_thread(collection.get)
            context_docs = results.get('documents', []) if results else []
        else:
            # Truy vấn Vector Search lấy top 3 sản phẩm phù hợp nhất
            results = await asyncio.to_thread(
                collection.query,
                query_texts=[chat.message],
                n_results=3
            )
            context_docs = results.get('documents', [[]])[0] if results and results.get('documents') else []
        
        # Lắp ráp dữ liệu sản phẩm tìm được
        context_text = "\n\n---\n\n".join(context_docs)
        
        # Nếu cửa hàng chưa có sản phẩm nào
        if not context_text:
            context_text = "Hiện tại cửa hàng chưa có thông tin sản phẩm nào."

        # 4. Tạo System Prompt (Nhắc nhở AI về vai trò của nó)
        system_prompt = f"""Bạn là trợ lý AI tư vấn bán hàng thân thiện, chuyên nghiệp và lịch sự của cửa hàng Chinh Hoops.
Nhiệm vụ của bạn là trả lời câu hỏi và tư vấn sản phẩm dựa vào thông tin của cửa hàng dưới đây.
Đặc biệt lưu ý về giá bán: Hãy luôn kiểm tra kỹ phần "Thông tin chi tiết các biến thể (Màu sắc, Kích cỡ & Tồn kho)" bên dưới để trả lời đúng giá của từng biến thể cụ thể (màu sắc, kích cỡ). Nếu biến thể có giá cụ thể, hãy báo đúng giá của biến thể đó cho khách hàng thay vì chỉ lấy giá chung của sản phẩm.
Tuyệt đối KHÔNG ĐƯỢC bịa đặt sản phẩm, giá cả, hoặc thông tin không có trong danh sách.
Nếu khách hàng hỏi ngoài lề không liên quan tới mua bán sản phẩm, hãy từ chối khéo léo.

THÔNG TIN CÁC SẢN PHẨM PHÙ HỢP TRONG CỬA HÀNG:
{context_text}
"""
        
        # 4. Gọi mô hình Gemini với chế độ Bất đồng bộ (Async) để tăng tốc I/O mạng
        async def stream_generator():
            try:
                response_stream = await client.aio.models.generate_content_stream(
                    model="gemini-3.5-flash",
                    contents=chat.message,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                    )
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
            except Exception as e:
                print(f"Gemini API Exception: {e}")
                yield "[Hệ thống]: Hiện tại máy chủ AI của Google đang quá tải (503 Service Unavailable) hoặc vượt quá hạn mức yêu cầu. Vui lòng gửi lại tin nhắn sau vài giây."
                    
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        
        # Trả về Stream thẳng xuống Frontend
        return StreamingResponse(stream_generator(), media_type="text/plain", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
