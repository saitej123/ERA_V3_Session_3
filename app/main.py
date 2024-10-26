from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import io
import pandas as pd
import cv2
import numpy as np
import base64
from .preprocessing import preprocess_text, clean_csv, perform_ner, perform_topic_modeling, classify_text, process_audio
import json
import librosa

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=FileResponse)
async def read_root():
    return FileResponse("templates/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        return {"sample": df.head().to_html(), "type": "csv"}
    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = process_image(contents)
        return {"image": image, "type": "image"}
    elif file.filename.lower().endswith(('.wav', '.mp3')):
        audio_data = process_audio(contents)
        return {"audio": audio_data, "type": "audio"}
    else:
        text = contents.decode("utf-8")
        return {"sample": text[:1000], "type": "text"}

@app.post("/preprocess")
async def preprocess(text: str = Form(...)):
    processed_text = preprocess_text(text)
    return {"processed": processed_text[:1000]}

@app.post("/clean_csv")
async def clean_csv_endpoint(data: str = Form(...)):
    df = pd.read_json(data)
    cleaned_df = clean_csv(df)
    return {"cleaned": cleaned_df.to_html()}

@app.post("/ner")
async def ner(text: str = Form(...)):
    entities = perform_ner(text)
    return {"entities": json.dumps(entities)}

@app.post("/topic_modeling")
async def topic_modeling(text: str = Form(...)):
    topics = perform_topic_modeling(text)
    return {"topics": topics}

@app.post("/classify")
async def classify(text: str = Form(...)):
    classification = classify_text(text)
    return {"classification": classification}

@app.post("/process_image")
async def process_image_endpoint(file: UploadFile = File(...), operation: str = Form(...)):
    contents = await file.read()
    processed_image = process_image(contents, operation)
    return JSONResponse(content={"image": processed_image})

@app.post("/process_audio")
async def process_audio_endpoint(file: UploadFile = File(...), operation: str = Form(...)):
    contents = await file.read()
    processed_audio = process_audio(contents, operation)
    return JSONResponse(content={"audio": processed_audio})

def process_image(contents, operation='original'):
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if operation == 'grayscale':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == 'edge_detection':
        img = cv2.Canny(img, 100, 200)
    elif operation == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
    elif operation == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        img = cv2.transform(img, kernel)
    elif operation == 'invert':
        img = cv2.bitwise_not(img)
    elif operation == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        gauss = cv2.GaussianBlur(inv, ksize=(15, 15), sigmaX=0, sigmaY=0)
        img = cv2.divide(gray, 255 - gauss, scale=256)
    
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def process_audio(contents, operation='original'):
    y, sr = librosa.load(io.BytesIO(contents))
    
    if operation == 'noise_reduction':
        y = librosa.effects.preemphasis(y)
    elif operation == 'pitch_shift':
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif operation == 'time_stretch':
        y = librosa.effects.time_stretch(y, rate=1.2)
    elif operation == 'reverb':
        y = np.concatenate([y, librosa.effects.preemphasis(y)])
    
    audio_bytes = librosa.util.buf_to_float(y)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_base64
