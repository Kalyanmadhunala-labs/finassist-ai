import os
import logging

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder


load_dotenv()


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PostgreSQL connection
# ---------------------------------------------------------

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)


# ---------------------------------------------------------
# Embedding + reranking models
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------
# Groq / OpenAI-compatible client
# ---------------------------------------------------------

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ---------------------------------------------------------
# ROUTER
# ---------------------------------------------------------

def route_question(question):

    prompt = f"""
You are the routing component of an enterprise AI system.

Classify the user's question into exactly one category:

SQL
RAG
BOTH

Choose SQL when the question requires structured business
data, calculations, aggregations, filtering, grouping, or
information from the sales database.

Choose RAG when the question requires information from
documents.

Choose BOTH when the question requires information from
both structured business data and documents.

Return only one word:
SQL
RAG
or
BOTH

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = (
        response
        .choices[0]
        .message
        .content
        .strip()
        .upper()
    )

    if result not in ["SQL", "RAG", "BOTH"]:
        return "RAG"

    return result


# ---------------------------------------------------------
# SQL TOOL
# ---------------------------------------------------------

def run_sql_question(question):

    prompt = f"""
You are a PostgreSQL SQL generation engine.

Your job is to generate exactly ONE SELECT query
for the structured business data.

Database table:

sales_transactions(
    transaction_id,
    transaction_date,
    region,
    product,
    customer_segment,
    quantity,
    revenue
)

Rules:

1. Return ONLY the SQL query.
2. Use ONLY the sales_transactions table.
3. Never modify the database.
4. Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   TRUNCATE, GRANT, REVOKE, or other write/DDL operations.
5. For revenue questions, use the revenue column directly.
6. Do NOT multiply revenue by quantity unless the user
   explicitly asks for a calculated value based on quantity.
7. For "total revenue", use SUM(revenue).
8. Filter the requested region, product, or customer segment exactly.
9. Do not invent data.
10. Do not add calculations that were not requested.
11. Return exactly one PostgreSQL SELECT statement.

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate safe and accurate "
                    "PostgreSQL SELECT queries."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    query = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    query = query.replace("```sql", "")
    query = query.replace("```", "")
    query = query.strip()

    logger.info("Generated SQL query: %s", query)

    query_lower = query.lower()

    blocked_keywords = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create",
        "grant",
        "revoke",
        "replace"
    ]

    if not query_lower.startswith("select"):

        logger.error(
            "Rejected non-SELECT SQL query."
        )

        return {
            "status": "ERROR",
            "message": (
                "Generated SQL query is not "
                "a SELECT statement."
            )
        }

    if ";" in query_lower[:-1]:

        logger.error(
            "Rejected SQL containing multiple statements."
        )

        return {
            "status": "ERROR",
            "message": (
                "Multiple SQL statements "
                "are not allowed."
            )
        }

    for keyword in blocked_keywords:

        if keyword in query_lower:

            logger.error(
                "Rejected SQL containing blocked keyword: %s",
                keyword
            )

            return {
                "status": "ERROR",
                "message": (
                    f"Blocked SQL keyword detected: {keyword}"
                )
            }

    # -----------------------------------------------------
    # Execute SQL
    # -----------------------------------------------------

    with conn.cursor() as cur:

        cur.execute(query)

        results = cur.fetchall()

    logger.info(
        "Raw SQL database result: %s",
        results
    )

    if not results:

        logger.info(
            "SQL returned no matching records."
        )

        return {
            "status": "NO_DATA",
            "result": []
        }

    return {
        "status": "DATA_FOUND",
        "result": results
    }


# ---------------------------------------------------------
# RAG RETRIEVAL
# ---------------------------------------------------------

def retrieve_rag_context(question):

    question_embedding = embedding_model.encode(
        question
    ).astype("float32")

    with conn.cursor() as cur:

        cur.execute(
            """
            SELECT
                chunk_id,
                content
            FROM document_chunks
            ORDER BY embedding <=> %s
            LIMIT 5
            """,
            (question_embedding,)
        )

        vector_results = cur.fetchall()

    if not vector_results:

        return ""

    pairs = []

    for chunk_id, content in vector_results:

        pairs.append(
            [question, content]
        )

    scores = reranker.predict(
        pairs
    )

    ranked_results = []

    for i, result in enumerate(vector_results):

        chunk_id, content = result

        ranked_results.append(
            (
                chunk_id,
                content,
                float(scores[i])
            )
        )

    ranked_results.sort(
        key=lambda x: x[2],
        reverse=True
    )

    top_results = ranked_results[:2]

    context_parts = []

    for chunk_id, content, score in top_results:

        context_parts.append(
            content
        )

    return "\n\n".join(
        context_parts
    )


# ---------------------------------------------------------
# RAG ANSWER
# ---------------------------------------------------------

def generate_rag_answer(
    question,
    context
):

    prompt = f"""
Answer the user's question using only the context.

If the answer is not available in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ---------------------------------------------------------
# SQL ANSWER
# ---------------------------------------------------------

def generate_sql_answer(
    question,
    sql_results
):

    prompt = f"""
Answer the user's business question using the SQL result.

The revenue values are in Indian Rupees (INR).
Use the ₹ symbol when presenting revenue.

Important:

- If the SQL result status is NO_DATA, clearly state
  that no matching records were found.
- Do not convert NO_DATA into ₹0.
- Do not invent a value.
- If the SQL result status is ERROR, explain that the
  database query could not be executed.
- Use the SQL result as the source of truth.
- Do not perform additional calculations unless
  explicitly required by the question.

Question:
{question}

SQL result:
{sql_results}

Give a clear, concise business answer.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


# ---------------------------------------------------------
# COMBINED SQL + RAG ANSWER
# ---------------------------------------------------------

def generate_combined_answer(
    question,
    sql_results,
    rag_context
):

    prompt = f"""
You are the final synthesis component of an enterprise
AI decision intelligence system.

Answer the user's question using BOTH sources below.

Structured business data from PostgreSQL:
{sql_results}

Information retrieved from internal documents:
{rag_context}

Rules:

- Use the PostgreSQL result for numerical business facts.
- Use the document context for document-based information.
- Do not invent information.
- If the SQL status is NO_DATA, say that no matching
  records were found.
- Do not report ₹0 unless the data explicitly shows zero revenue.
- If one source does not contain the required information,
  clearly say that it is not available.
- Revenue values are in Indian Rupees (INR).
- Use the ₹ symbol for revenue.
- Do not perform additional calculations that are not
  supported by the source data.
- Combine the information into one clear business answer.

User question:
{question}

Final answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )