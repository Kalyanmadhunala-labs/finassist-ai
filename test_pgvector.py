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

# Enable pgvector support for this Python connection
register_vector(conn)

# Create an embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Machine Learning is a subset of Artificial Intelligence."

embedding = model.encode(text)

# Insert one test document
with conn.cursor() as cur:
    cur.execute(
        """
        INSERT INTO document_chunks
        (chunk_id, source, content, embedding)
        VALUES (%s, %s, %s, %s)
        """,
        (999, "test.txt", text, embedding)
    )

conn.commit()

print("Test vector inserted successfully!")

conn.close()