from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

class AgentInput(BaseModel):
    question: str
    user_id: int

class AgentOutput(BaseModel):
    response: str
    success: bool

    def __init__(self):
        self.model_name = "SimpleAgent"
    
    def process_input(self, input_data: dict) -> dict:
        try:
            # Kiểm tra input bằng BaseModel
            agent_input = AgentInput(**input_data)
            print(f"Input is valid for {self.model_name}.")

            # Xử lý logic tại đây (ví dụ, phản hồi dựa trên câu hỏi)
            response = self.generate_response(agent_input.question)
            
            # Tạo output
            agent_output = AgentOutput(response=response, success=True)
            return agent_output.dict()
        
        except ValidationError as e:
            # Xử lý lỗi nếu input không hợp lệ
            return {"response": f"Invalid input: {e}", "success": False}
    
    def generate_response(self, question: str) -> AgentOutput:
        # Logic trả lời đơn giản (ví dụ)
        if "hello" in question.lower():
            return "Hello! How can I assist you today?"
        else:
            return "I'm not sure how to answer that."

# Sử dụng agent
input_data = {
    "question": "Hello, agent!",
    "user_id": 12345
}

# Khởi tạo agent và xử lý input
agent = SimpleAgent()
output_data = agent.process_input(input_data)

# In kết quả
print(output_data)


