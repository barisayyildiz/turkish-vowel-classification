from typing import Union

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# from tensorflow.keras.optimizers import SGD
# from keras.preprocessing.image import ImageDataGenerator


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

@app.post("/analyze")
async def anaylze_sound(sound: UploadFile = File()):
	# print(sound)
	content_sound = await sound.read()
	print(sound)
	print(sound.content_type)
	print(type(content_sound))
	print(len(content_sound))
	print(sound.filename)

	print(sound.file)

	rnd_id = str(uuid.uuid4())

	root_dir = "audios"
	path = os.path.join(root_dir, rnd_id + ".wav")
	# print(path)

	with open(path, mode="wb") as f:
		f.write(content_sound)
	audio_to_spec(path)
	prediction = predict_spec()

	remove_file(rnd_id)



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


# def main():
# 	filename ="vowels_spec_model"
# 	model_s = get_loaded_model_by_name(filename)
# 	opt = SGD(lr=0.001)
# 	model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'], run_eagerly=True)

# 	print(model_s)

# 	################ Spectrogram images ##############
# 	h = 96
# 	w = 96

# 	path_spec_test = "specs"
# 	spec_datagen_test = ImageDataGenerator()

# 	test_generator_spec = spec_datagen_test.flow_from_directory(path_spec_test,class_mode='categorical',color_mode ='rgb',shuffle = False,  target_size =(h,w),subset='training')
# 	test_spec_data,test_labels_spec = load_data_from_generators(test_generator_spec,timestep=1,color_mode='rgb')


# main()

