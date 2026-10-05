import time

from fastapi import FastAPI
from pydantic import BaseModel

from finassist_graph import app


api = FastAPI(
    title="Finassist AI API",
    description="Enterprise AI Decision Intelligence API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    response_time_seconds: float


@api.get("/")
def home():
    return {
        "message": "Finassist AI API is running"
    }


@api.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    start_time = time.time()

    result = app.invoke(
        {
            "question": request.question,
            "route": "",
            "sql_result": {},
            "rag_result": "",
            "validation_result": False,
            "final_answer": ""
        }
    )

    elapsed_time = time.time() - start_time

    return {
        "question": request.question,
        "answer": result["final_answer"],
        "response_time_seconds": round(elapsed_time, 2)
    }