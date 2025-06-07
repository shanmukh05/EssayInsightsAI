from pydantic import BaseModel
from typing import Optional, List


class ChatRequest(BaseModel):
    chat_history: List[dict]
    user_query: str
    essay_text: str
    essay_analysis: str
    model: Optional[str] = "gpt-4o"
    openai_key: str


class ChatResponse(BaseModel):
    chat_reply: str
