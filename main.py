import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langflow import load_flow_from_json
import uvicorn

FLOW_PATH = "Italian_Grandma_Food_Coach.json"

with open(FLOW_PATH, "r") as f:
    flow_data = json.load(f)

flow = load_flow_from_json(flow_data)

app = FastAPI()

# Allow CORS for testing/embedding in Wix
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your Wix domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await flow.predict(input_value=req.message)
        return {"response": result}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
