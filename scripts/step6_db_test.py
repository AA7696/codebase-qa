import os

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)

    conn.execute("DROP TABLE IF EXISTS vector_test")
    conn.execute("CREATE TABLE vector_test (id serial PRIMARY KEY, label text, embedding vector(3))")

    rows = [
        ("near x axis", np.array([1.0, 0.1, 0.0])),
        ("near y axis", np.array([0.1, 1.0, 0.0])),
        ("near z axis", np.array([0.0, 0.1, 1.0])),
    ]
    for label, vector in rows:
        conn.execute("INSERT INTO vector_test (label, embedding) VALUES (%s, %s)", (label, vector))

    query = np.array([0.9, 0.2, 0.0])
    result = conn.execute(
        "SELECT label, embedding <=> %s AS distance FROM vector_test ORDER BY distance LIMIT 3",
        (query,),
    ).fetchall()
    for label, distance in result:
        print(f"{distance:.3f}  {label}")

    conn.execute("DROP TABLE vector_test")