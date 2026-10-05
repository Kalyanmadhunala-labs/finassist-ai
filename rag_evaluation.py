import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to PostgreSQL
conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)

print("Connected to PostgreSQL")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Evaluation questions
evaluation_data = [
    {
        "question": "What is Machine Learning?",
        "expected_chunk": 1
    },
    {
        "question": "What is Generative AI?",
        "expected_chunk": 2
    },
    {
        "question": "What are Large Language Models?",
        "expected_chunk": 3
    },
    {
        "question": "What is RAG?",
        "expected_chunk": 3
    },
    {
        "question": "What is chunking?",
        "expected_chunk": 4
    }
]

top_k = 2

correct = 0

for item in evaluation_data:

    question = item["question"]
    expected_chunk = item["expected_chunk"]

    question_embedding = model.encode(question).astype("float32")

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT chunk_id, content,
                   1 - (embedding <=> %s) AS similarity
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                question_embedding,
                question_embedding,
                top_k
            )
        )

        results = cur.fetchall()

    retrieved_chunks = [row[0] for row in results]

    is_correct = expected_chunk in retrieved_chunks

    if is_correct:
        correct += 1

    print("\nQuestion:", question)
    print("Expected chunk:", expected_chunk)
    print("Retrieved chunks:", retrieved_chunks)
    print("Correct:", is_correct)


accuracy = correct / len(evaluation_data)

print("\n-----------------------------")
print("Retrieval Evaluation")
print("-----------------------------")
print("Correct:", correct)
print("Total:", len(evaluation_data))
print("Top-K Retrieval Accuracy:", accuracy)

conn.close()