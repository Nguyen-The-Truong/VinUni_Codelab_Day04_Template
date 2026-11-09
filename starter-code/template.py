"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng cho VinFast, Vinpearl và các thương hiệu Vingroup.
- Giọng nói: Chuyên nghiệp, rõ ràng, thân thiện, chính xác.

## AVAILABLE TOOLS
- search_product_catalog(category: str, max_price: int = 999999999999): Tìm sản phẩm/dịch vụ trong danh mục theo loại và mức giá tối đa.
- submit_support_ticket(customer_name: str, issue_description: str, priority: str = "medium"): Tạo ticket hỗ trợ cho khách hàng khi cần xử lý sự cố.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu về sản phẩm, giá, chính sách, hoặc hỗ trợ khách hàng.
2. Nếu người dùng cần thông tin thực tế, phải gọi tool trước khi trả lời.
3. Nếu tool không trả về kết quả, nói rõ và đề xuất cách xử lý tiếp theo.
4. Luôn giữ câu trả lời tập trung vào Vingroup và sản phẩm/dịch vụ thực của hệ sinh thái.
5. Chỉ trả lời dựa trên dữ liệu đã có, không suy đoán ngoài phạm vi cho phép.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời về VinFast, Vinpearl, dịch vụ Vingroup, và sản phẩm trong catalog đã cho.
- Không trả lời các chủ đề ngoài phạm vi Vingroup.
- Khi được yêu cầu tạo ticket, cần xác định rõ tên khách hàng, vấn đề, và mức ưu tiên.

## OUTPUT CONTRACT
Trả lời theo cấu trúc:
Thought: Tóm tắt ý định và cần dùng tool nào.
Action: Gọi tool tương ứng nếu cần.
Observation: Dữ liệu trả về từ tool.
Final Answer: Kết luận ngắn gọn, dễ đọc, dựa trên dữ liệu thực.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Mô phỏng chatbot không dùng tool, chỉ trả lời trực tiếp.
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _extract_category(self, user_input: str) -> str:
        text = user_input.lower()
        if "du lịch" in text or "tour" in text or "resort" in text or "phú quốc" in text or "nhatrang" in text:
            return "du_lich"
        return "xe_dien"

    def _extract_price_limit(self, user_input: str) -> int:
        match = re.search(r"dưới\s*(\d+(?:\.\d+)?)\s*(triệu|trieu|m|vnđ|vnd)?", user_input.lower())
        if match:
            value = float(match.group(1))
            if "triệu" in user_input.lower() or "trieu" in user_input.lower() or "m" in user_input.lower():
                return int(value * 1000000)
        return 999999999999

    def _detect_intent(self, user_input: str):
        text = user_input.lower()
        ticket_markers = [
            "lỗi", "hỗ trợ", "ticket", "vấn đề", "khách hàng", "báo lỗi", "gấp", "nghiêm trọng", "adas", "hệ thống", "tên"
        ]
        catalog_markers = [
            "xe", "xe điện", "vinfast", "du lịch", "tour", "resort", "mua", "xem", "giá", "dưới", "sản phẩm"
        ]
        is_faq = any(k in text for k in ["bảo hành", "chính sách", "bao lâu", "hướng dẫn", "faq"])

        needs_ticket = any(k in text for k in ticket_markers)
        needs_catalog = any(k in text for k in catalog_markers)

        if needs_ticket and not re.search(r"\b(?:xem|mua|tìm|giá|dưới|sản phẩm|catalog|du lịch|tour|resort)\b", text):
            needs_catalog = False
        if "du lịch" in text or "tour" in text or "resort" in text:
            needs_catalog = True

        if is_faq:
            needs_catalog = False
            needs_ticket = False

        return {
            "needs_catalog": needs_catalog and not is_faq,
            "needs_ticket": needs_ticket and not is_faq,
            "is_faq": is_faq
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)
        self.trace.append({"step": "intent", "user_input": user_input, **intents})

        iteration = 1
        while iteration <= self.max_iterations:
            if intents["needs_catalog"]:
                category = self._extract_category(user_input)
                max_price = self._extract_price_limit(user_input)
                results = search_product_catalog(category=category, max_price=max_price)
                self.trace.append({
                    "step": "catalog_search",
                    "category": category,
                    "max_price": max_price,
                    "results": results
                })

                if not results or (len(results) == 1 and "error" in results[0]):
                    if intents["needs_ticket"]:
                        intents["needs_catalog"] = False
                    else:
                        answer = "Rất tiếc, không tìm thấy sản phẩm phù hợp trong khoảng giá này."
                        return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}

                if intents["needs_catalog"]:
                    names = ", ".join(item.get("name", "") for item in results[:2])
                    answer = f"Tôi tìm thấy các sản phẩm phù hợp: {names}."
                    if not intents["needs_ticket"]:
                        return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}
                    intents["needs_catalog"] = False
                    self.trace.append({"step": "catalog_summary", "answer": answer})
                    iteration += 1
                    continue

            if intents["needs_ticket"]:
                customer_name_match = re.search(r"tôi tên\s+([^,]+)", user_input, flags=re.IGNORECASE)
                customer_name = customer_name_match.group(1).strip() if customer_name_match else "Khách hàng"
                issue_description = user_input.strip()
                priority = "high" if any(k in user_input.lower() for k in ["nghiêm trọng", "gấp", "khẩn cấp", "khẩn", "bị lỗi", "lỗi hệ thống"]) else "medium"
                ticket = submit_support_ticket(
                    customer_name=customer_name,
                    issue_description=issue_description,
                    priority=priority
                )
                self.trace.append({"step": "ticket_submit", "ticket": ticket})
                answer = f"Tôi đã tạo ticket cho {customer_name} với mã {ticket['ticket_id']} ({ticket['status']})."
                return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}

            if intents["is_faq"]:
                answer = "Theo chính sách của VinFast, pin xe điện được bảo hành lên đến 10 năm, tùy thuộc vào từng dòng xe và điều kiện bảo hành cụ thể."
                self.trace.append({"step": "faq_response", "answer": answer})
                return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}

            answer = "Tôi có thể hỗ trợ bạn về sản phẩm VinFast, du lịch Vinpearl hoặc tạo ticket hỗ trợ khách hàng."
            self.trace.append({"step": "fallback_response", "answer": answer})
            return {"answer": answer, "trace": self.trace, "iterations": iteration, "status": "completed"}

        return {
            "answer": "Lỗi: Vượt quá số bước tối đa.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
