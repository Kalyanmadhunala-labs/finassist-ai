import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


def run_sql(query, parameters=None):

    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    try:
        with conn.cursor() as cur:

            cur.execute(
                query,
                parameters
            )

            results = cur.fetchall()

        return results

    finally:
        conn.close()