import string
from tokenize import String
from fastapi import FastAPI
from pydantic import BaseModel


import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

# Load model and scaler
with open('ml_models/tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

loaded_model = tf.keras.models.load_model('ml_models')


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/classify/")
def classify(str:string):
    input = pad_sequences(loaded_tokenizer.texts_to_sequences([str]),maxlen = 40)
    res = loaded_model.predict(input)
    if res[0][0] > 0.55:
        res = 'Positive'
    elif  res[0][0] < 0.45:
        res = 'Negative'
    else:
        res = 'Neutral'
    
    return {'Result': res}