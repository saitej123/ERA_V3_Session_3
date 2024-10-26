import re
import spacy
import pandas as pd
from transformers import pipeline
import librosa
import numpy as np
import io

nlp = spacy.load("en_core_web_md")

# Initialize zero-shot classification pipeline
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def preprocess_text(text):
    doc = nlp(text)
    
    # Tokenization and lemmatization
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    # Remove special characters and digits
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    
    # Remove extra whitespace
    processed_text = ' '.join(tokens)
    
    return processed_text

def clean_csv(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna('Unknown')
    
    # Convert text columns to lowercase
    text_columns = df.select_dtypes(include=['object']).columns
    df[text_columns] = df[text_columns].apply(lambda x: x.str.lower())
    
    return df

def perform_ner(text):
    ner_results = ner_pipeline(text)
    entities = [(entity['word'], entity['entity']) for entity in ner_results]
    return entities

def perform_topic_modeling(text, num_topics=3):
    # Define potential topics
    topics = ["Business", "Technology", "Politics", "Science", "Entertainment", "Sports", "Health"]
    
    # Perform zero-shot classification for topic modeling
    result = zero_shot_classifier(text, topics, multi_label=True)
    
    # Sort and select top 'num_topics' topics
    top_topics = sorted(zip(result['labels'], result['scores']), key=lambda x: x[1], reverse=True)[:num_topics]
    
    return [f"Topic: {topic}, Score: {score:.2f}" for topic, score in top_topics]

def classify_text(text, categories=['business', 'sports', 'technology', 'entertainment']):
    result = zero_shot_classifier(text, categories)
    return f"Classification: {result['labels'][0]}, Score: {result['scores'][0]:.2f}"

def process_audio(audio_data, operation='original'):
    y, sr = librosa.load(io.BytesIO(audio_data))
    
    if operation == 'noise_reduction':
        y = librosa.effects.preemphasis(y)
    elif operation == 'pitch_shift':
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    elif operation == 'time_stretch':
        y = librosa.effects.time_stretch(y, rate=1.2)
    elif operation == 'reverb':
        y = np.concatenate([y, librosa.effects.preemphasis(y)])
    
    return y, sr
