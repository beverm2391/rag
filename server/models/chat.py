from typing import List, Optional
from pydantic import BaseModel, Field

# ! Requests Models ========================
class Message(BaseModel):
    role: str
    content: Optional[str]

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    model: Optional[str] = Field(default=None)

# ! Response Models ========================
class ChatResponseData(BaseModel):
    text: str

class ChatResponse(BaseModel):
    type: str
    data: ChatResponseData
