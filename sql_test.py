import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)


print("Connected to PostgreSQL")


with conn.cursor() as cur:

    cur.execute(
        """
        SELECT SUM(revenue)
        FROM sales_transactions
        WHERE region = %s
        """,
        ("South",)
    )

    result = cur.fetchone()


print("South region revenue:", result[0])


conn.close()

print("Database connection closed")