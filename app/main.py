from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import io
import pandas as pd
import cv2
import numpy as np
import base64
from .preprocessing import preprocess_text, clean_csv, perform_ner, perform_topic_modeling, classify_text
from .audio_processing import process_audio
import json
import librosa
import typing

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(None), text: str = Form(None)):
    if file:
        contents = await file.read()
        if file.filename and file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            return {"sample": df.head().to_html(), "type": "csv"}
        elif file.filename and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = process_image(contents)
            return {"image": image, "type": "image"}
        elif file.filename and file.filename.lower().endswith(('.wav', '.mp3')):
            audio_data = process_audio(contents)
            return {"audio": audio_data, "type": "audio"}
        else:
            text = contents.decode("utf-8")
    
    if text:
        return {"sample": text[:1000], "type": "text"}
    
    return {"error": "No file or text provided"}

@app.post("/upload_camera")
async def upload_camera(image: str = Form(...)):
    image_data = base64.b64decode(image.split(',')[1])
    processed_image = process_image(image_data)
    return {"image": processed_image, "type": "image"}

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_audio = process_audio(contents)
        return {"audio": processed_audio, "type": "audio"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.post("/process_audio")
async def process_audio_endpoint(file: UploadFile = File(...), operation: str = Form(...)):
    try:
        contents = await file.read()
        processed_audio = process_audio(contents, operation)
        return JSONResponse(content={"audio": processed_audio})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

@app.post("/text_clean")
async def text_clean(text: str = Form(...)):
    processed_text = preprocess_text(text)
    return {"cleaned_text": processed_text[:1000]}

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

@app.post("/preprocess")
async def preprocess(text: str = Form(...)):
    processed_text = preprocess_text(text)
    return {"processed": processed_text[:1000]}

def process_image(contents: bytes, operation: str = 'original') -> str:
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
    return base64.b64encode(buffer.tobytes()).decode('utf-8')
