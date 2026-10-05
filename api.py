import time

from fastapi import FastAPI
from pydantic import BaseModel, Field

from finassist_graph import run_finassist


api = FastAPI(
    title="Finassist AI API",
    description="Enterprise AI Decision Intelligence API",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User question for Finassist AI"
    )


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

    # Reject whitespace-only questions.
    if not request.question.strip():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    start_time = time.time()

    result = run_finassist(
        {
            "question": request.question.strip(),
            "route": "",
            "sql_result": {},
            "rag_result": "",
            "validation_result": False,
            "final_answer": ""
        }
    )

    elapsed = time.time() - start_time

    return {
        "question": request.question.strip(),
        "answer": result["final_answer"],
        "response_time_seconds": round(elapsed, 2)
    }