from fastapi import FastAPI
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from chatbot.backend.schemas.request_response import ChatRequest, ChatResponse
from chatbot.backend.services.assistant import generate_chat_reply

app = FastAPI()


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    reply = generate_chat_reply(
        chat_history=request.chat_history,
        user_query=request.user_query,
        essay_text=request.essay_text,
        essay_analysis=request.essay_analysis,
        model=request.model,
        openai_key=request.openai_key,
    )
    return ChatResponse(chat_reply=reply)
