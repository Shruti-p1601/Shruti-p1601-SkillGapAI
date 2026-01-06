from fastapi import FastAPI
from extractor import extract_skills


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Server running"}

@app.post("/extract")
def extract(data: dict):
    text = data["text"]
    skills = extract_skills(text)
    return {"skills": skills}

