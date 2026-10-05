import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CONNECT TO POSTGRESQL
# ============================================================

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)

print("Connected to PostgreSQL")


# ============================================================
# 3. LOAD MODELS
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Models loaded")


# ============================================================
# 4. EVALUATION DATASET
# ============================================================

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
        "expected_chunk": 2
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


# ============================================================
# 5. SETTINGS
# ============================================================

candidate_limit = 5
top_k = 2

correct = 0


# ============================================================
# 6. EVALUATE EACH QUESTION
# ============================================================

for item in evaluation_data:

    question = item["question"]
    expected_chunk = item["expected_chunk"]

    print("\n========================================")
    print("Question:", question)
    print("Expected chunk:", expected_chunk)
    print("========================================")


    # --------------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        question
    ).astype("float32")


    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                chunk_id,
                content
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (
                question_embedding,
                candidate_limit
            )
        )

        vector_results = cur.fetchall()


    # --------------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------------

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                chunk_id,
                content
            FROM document_chunks
            WHERE search_vector @@
                  plainto_tsquery('english', %s)
            LIMIT %s
            """,
            (
                question,
                candidate_limit
            )
        )

        keyword_results = cur.fetchall()


    # --------------------------------------------------------
    # COMBINE CANDIDATES
    # --------------------------------------------------------

    candidates = {}

    for chunk_id, content in vector_results:

        candidates[chunk_id] = content


    for chunk_id, content in keyword_results:

        candidates[chunk_id] = content


    # --------------------------------------------------------
    # CREATE QUESTION-CHUNK PAIRS
    # --------------------------------------------------------

    pairs = []

    candidate_items = list(
        candidates.items()
    )

    for chunk_id, content in candidate_items:

        pairs.append(
            [question, content]
        )


    # --------------------------------------------------------
    # RERANK
    # --------------------------------------------------------

    reranker_scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------------
    # ATTACH SCORES
    # --------------------------------------------------------

    ranked_results = []

    for i, (chunk_id, content) in enumerate(
        candidate_items
    ):

        score = float(
            reranker_scores[i]
        )

        ranked_results.append(
            (
                chunk_id,
                content,
                score
            )
        )


    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    ranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )


    # --------------------------------------------------------
    # SELECT TOP K
    # --------------------------------------------------------

    top_results = ranked_results[:top_k]

    retrieved_chunks = [
        chunk_id
        for chunk_id, content, score
        in top_results
    ]


    # --------------------------------------------------------
    # CHECK RESULT
    # --------------------------------------------------------

    is_correct = (
        expected_chunk
        in retrieved_chunks
    )


    if is_correct:

        correct += 1


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        "Retrieved chunks:",
        retrieved_chunks
    )

    print(
        "Correct:",
        is_correct
    )


# ============================================================
# 7. FINAL METRIC
# ============================================================

total = len(evaluation_data)

accuracy = correct / total


print("\n========================================")
print("RAG EVALUATION RESULTS")
print("========================================")

print("Correct:", correct)

print("Total:", total)

print(
    "Top-K Retrieval Accuracy:",
    f"{accuracy:.2%}"
)


# ============================================================
# 8. CLOSE DATABASE
# ============================================================

conn.close()

print("\nDatabase connection closed.")