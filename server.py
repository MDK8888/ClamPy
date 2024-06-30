import time
import torch
import uvicorn
from fastapi.responses import FileResponse
from TTS.api import TTS
from fastapi import FastAPI
from fast API.middleware.cors import CORSMiddleware

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



if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)


app = FastAPI()

origins = ["*"]  # fix later

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


@app.post("/tts")
async def tts():
    file_path = "output.wav"

    # Text to speech to a file
    tts.tts_to_file(text="This is a new file", file_path=file_path)

    # Return the wav file as a response
    return FileResponse(path=file_path, media_type="audio/wav", filename=file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
