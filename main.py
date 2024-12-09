from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Define the base URL for the LLaMA API
ollama_base_url = "http://localhost:11434/v1"

# Function to interact with Ollama's API (LLaMA-based)
def call_llama_api(prompt):
    headers = {
        "Authorization": f"Bearer {os.getenv('http://localhost:11434')}",  # Optional: Add your API key if needed
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "llama3.2",  
        "input": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(f"{ollama_base_url}/chat/completions", json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define the Pydantic model for input data
class InputData(BaseModel):
    text: str

# Define a POST endpoint for predictions
@app.post("/predict")
async def predict(input_data: InputData):
    model_input = input_data.text
    model_response = call_llama_api(model_input)
    
    return {"prediction": model_response}
