from fastapi import FastAPI, HTTPException
import requests
import os
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    text: str

ollama_base_url = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

@app.post("/generate")
async def generate_response(prompt: Prompt):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": prompt.text}
        ]
    }

    try:
        response = requests.post(f"{ollama_base_url}/chat/completions", json=data, headers=headers)
        response.raise_for_status()
        return {"response": response.json()['choices'][0]['message']['content']}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
