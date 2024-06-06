from fastapi import FastAPI
import psycopg2
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
    cursor = connect.cursor()
    cursor.execute(
        """
        SELECT *
        FROM cd.bookings
        """
)
    return cursor.fetchall()