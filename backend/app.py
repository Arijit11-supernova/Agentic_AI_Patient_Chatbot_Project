from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "Backend is LIVE on Vercel"}







