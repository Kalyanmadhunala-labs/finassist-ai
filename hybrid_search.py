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

question = "What is Generative AI?"

print("\nQuestion:")
print(question)

# Convert question into an embedding
question_embedding = model.encode(question).astype("float32")

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            chunk_id,
            content,

            1 - (embedding <=> %s) AS vector_score,

            ts_rank(
                search_vector,
                plainto_tsquery('english', %s)
            ) AS keyword_score

        FROM document_chunks

        WHERE search_vector @@
              plainto_tsquery('english', %s)

        ORDER BY vector_score DESC

        LIMIT 5
        """,
        (
            question_embedding,
            question,
            question
        )
    )

    results = cur.fetchall()


print("\nHybrid Search Results:")

for chunk_id, content, vector_score, keyword_score in results:

    print("\nChunk ID:", chunk_id)
    print("Vector score:", round(vector_score, 4))
    print("Keyword score:", round(keyword_score, 4))
    print("Content:", content)

conn.close()