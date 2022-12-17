from typing import Union

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import time

import io

from tempfile import NamedTemporaryFile
import shutil
from pathlib import Path

# creating spectogram
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from utils import audio_to_spec, get_loaded_model_by_name, load_data_from_generators, predict_spec, remove_file

import os
import uuid



# ABOUT MODAL
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img,array_to_img
from keras.preprocessing.image import DirectoryIterator,DataFrameIterator




class Item(BaseModel):
	name: str
	description: Union[str, None] = None
	price: float
	tax: Union[float, None] = None

origins = ["*"]

app = FastAPI()

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.get("/")
def read_root():
	return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
	return {"item_id": item_id, "q": q}

@app.post("/anaylze_test")
def deal_with_sound_file(sound: bytes = File(...)):
	file_bytes = io.BytesIO(sound)
	print(file_bytes)
	y, sr = librosa.load(file_bytes)

	S = librosa.feature.melspectrogram(y=y, sr=sr)

	cmap = plt.get_cmap('inferno')
	plt.figure(figsize=(8,8))

	plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
	plt.savefig(f'specs/images/test_file_name.png')

	return {"hello":"world"}

@app.post("/analyze")
async def anaylze_sound(sound: UploadFile = File()):
	start_time = time.time()

	content_sound = await sound.read()

	rnd_id = str(uuid.uuid4())
	root_dir = "audios"
	path = os.path.join(root_dir, rnd_id + ".wav")

	with open(path, mode="wb") as f:
		f.write(content_sound)

	print(f'duration : {librosa.get_duration(filename=root_dir + "/" + rnd_id + ".wav")}')

	audio_to_spec(path)
	prediction = predict_spec(model_s)
	remove_file()

	return {"vowel" : prediction}
	# return {"Hello": "World"}

@app.post("/save_upload_file_tmp")
def save_upload_file_tmp(file: UploadFile) -> Path:
	root_dir = "audios"
	path = os.path.join(root_dir, file.filename)
	with open(path, "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)

	audio_to_spec(path)
	
	return {"file_name":file.filename}


@app.post("/items/")
async def create_item(item: Item):
	print(item)
	item_dict = item.dict()
	if item.tax:
		price_with_tax = item.price + item.tax
		item_dict.update({"price_with_tax": price_with_tax})
	return item_dict


name = "barış"


# INITIAL LOADS
filename ="vowels_spec_model"
model_s = get_loaded_model_by_name(filename)
opt = SGD(lr=0.001)
model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'], run_eagerly=True)

