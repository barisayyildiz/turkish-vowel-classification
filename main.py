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

from utils import audio_to_spec

import os


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
	print(sound)
	content_sound = await sound.read()
	print(sound.filename)
	return {"Hello": "World"}

@app.post("/save_upload_file_tmp")
def save_upload_file_tmp(file: UploadFile) -> Path:
	root_dir = "audios"
	path = os.path.join(root_dir, file.filename)
	with open(path, "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)

	audio_to_spec(path)
	
	return {"file_name":file.filename}


	# try:
	# 	suffix = Path(upload_file.filename).suffix
	# 	with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
	# 		shutil.copyfileobj(upload_file.file, tmp)
	# 		tmp_path = Path(tmp.name)
	# finally:
	# 	print("hello")
	# 	upload_file.file.close()
	# return tmp_path


@app.post("/items/")
async def create_item(item: Item):
	print(item)
	item_dict = item.dict()
	if item.tax:
		price_with_tax = item.price + item.tax
		item_dict.update({"price_with_tax": price_with_tax})
	return item_dict


files = [i for i in os.listdir() if os.path.isfile(i)]
print(files)

