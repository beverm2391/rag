from typing import List, Optional
from pydantic import BaseModel

# ! Requests Models ========================
class Message(BaseModel):
    role: str
    content: Optional[str]

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int]
    temperature: Optional[float]
    model: Optional[str]

# ! Response Models ========================
class ChatResponseData(BaseModel):
    text: str

class ChatResponse(BaseModel):
    type: str
    data: ChatResponseData
