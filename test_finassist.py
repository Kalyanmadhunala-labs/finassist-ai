import requests

from finassist_graph import app


# ---------------------------------------------------------
# Test helper
# ---------------------------------------------------------

def run_question(question):
    result = app.invoke(
        {
            "question": question,
            "route": "",
            "sql_result": {},
            "rag_result": "",
            "validation_result": False,
            "final_answer": ""
        }
    )

    return result


# ---------------------------------------------------------
# TEST 1 — SQL
# ---------------------------------------------------------

def test_sql_question():

    question = "What is the total revenue in the South region?"

    result = run_question(question)

    assert result["route"] == "SQL"
    assert result["validation_result"] is True

    # Normalize commas before checking the amount.
    final_answer = result["final_answer"].replace(",", "")

    assert "1254000" in final_answer

    print("PASS: SQL question")


# ---------------------------------------------------------
# TEST 2 — RAG
# ---------------------------------------------------------

def test_rag_question():

    question = "What does our internal document say about RAG?"

    result = run_question(question)

    assert result["route"] == "RAG"
    assert result["validation_result"] is True
    assert result["rag_result"]

    print("PASS: RAG question")


# ---------------------------------------------------------
# TEST 3 — BOTH
# ---------------------------------------------------------

def test_both_question():

    question = (
        "What is the total revenue in the South region, "
        "and what does our internal document say about RAG?"
    )

    result = run_question(question)

    assert result["route"] == "BOTH"
    assert result["validation_result"] is True
    assert result["sql_result"]
    assert result["rag_result"]

    print("PASS: BOTH question")


# ---------------------------------------------------------
# TEST 4 — API health
# ---------------------------------------------------------

def test_api_health():

    response = requests.get(
        "http://127.0.0.1:8000/"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Finassist AI API is running"

    print("PASS: API health")


# ---------------------------------------------------------
# TEST 5 — API /ask
# ---------------------------------------------------------

def test_api_ask():

    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={
            "question": "What is the total revenue in the South region?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "What is the total revenue in the South region?"
    )

    assert data["answer"]
    assert data["response_time_seconds"] >= 0

    print("PASS: API /ask")


# ---------------------------------------------------------
# RUN ALL TESTS
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nRunning Finassist AI evaluation tests...\n")

    test_sql_question()
    test_rag_question()
    test_both_question()
    test_api_health()
    test_api_ask()

    print("\nAll Finassist AI tests passed successfully.")