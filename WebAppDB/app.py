import uvicorn
import dotenv
from fastapi import FastAPI
import psycopg2
import os
from psycopg2.extensions import connection
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

app = FastAPI()
@app.get("/")
def read_root():
    return {"Hello": "World"}

app.get("/sum")
def sum_two(a: int, b: int)-> int:
  return a + b

@app.get("/booking/all")
def all_bookings():
    conn = psycopg2.connect("/PATH/")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT *
        FROM cd.bookings
        """
)
    return cursor.fetchall()

if __name__ == "__main__":
    load_dotenv()
    uvicorn.run(app)