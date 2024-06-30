from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

origins = [
    "*" # fix later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def respond():
    time.sleep(1)
    return "I am the Golden Gate - Leo"

import torch as t 
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import pyttsx3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time 


app = FastAPI()

def generate_text_with_steering(prompt, steering_vector=None, coeff=1.0):
    """Generates text -  replace this with your *actual* model, SAE, 
        steering logic. 
    """ 
    generated_text = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return generated_text

class TextIn(BaseModel):
    text: str
    # Add steering parameters here if needed, e.g.:
    # steering_vector: List[float]  
    # coeff: float = 1.0 

@app.post("/generate_speech")
async def generate_speech(text_data: TextIn): 
    prompt = text_data.text
    
    # - (Optional) Retrieve Steering Parameters from Request -  
    # steering_vector =  t.tensor(text_data.steering_vector).to(DEVICE) 
    # coeff = text_data.coeff

    generated_text = generate_text_with_steering(prompt)  
    
    engine.say(generated_text)
    engine.runAndWait() 

    return {"message": f"Speech generated from: '{prompt}'", "generated_text":generated_text}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  
