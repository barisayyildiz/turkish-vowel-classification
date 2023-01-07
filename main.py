from typing import Union

from fastapi import FastAPI, Form, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import json

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

from utils import audio_to_spec, get_loaded_model_by_name, load_data_from_generators, predict_spec, remove_all, remove_file, predict_all

import os
import uuid

import soundfile as sf

from scipy import stats


# ABOUT MODAL
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
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

app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.post("/analyze-parts-new")
async def anaylze_sound_parts(sound: UploadFile = File()):
	try:
		start = time.time()

		content_sound = await sound.read()

		print('spend to read sound file: ', time.time() - start)
		start = time.time()

		rnd_id = str(uuid.uuid4())
		root_dir = "audios"
		path = os.path.join(root_dir, rnd_id + ".wav")

		with open(path, mode="wb") as f:
			f.write(content_sound)

		print('spend to write audio file: ', time.time() - start)
		start = time.time()

		print(f'duration : {librosa.get_duration(filename=root_dir + "/" + rnd_id + ".wav")}')

		wave, sr = librosa.load(os.path.join(root_dir, rnd_id + ".wav"), sr=None)

		segment_dur_secs = 0.03
		segment_length = int(sr * segment_dur_secs)

		num_sections = int(np.ceil(len(wave) / segment_length))
		split = []

		for i in range(num_sections):
			t = wave[i * segment_length: (i + 1) * segment_length]
			split.append(t)

		for i in range(num_sections):
			recording_name = os.path.basename("test")
			# out_file = f"{recording_name}_{str(i)}.wav"
			out_filename = "audios/split" + str(i) + "_.wav"
			sf.write(out_filename, split[i], sr)
			audio_to_spec(out_filename)
		
		predictions = predict_all(model_s)

		vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]

		x_data = np.arange(-5, 5, 0.001)
		y_data = stats.norm.pdf(x_data, 0, 1)
		n = len(predictions)

		res = {
			"a":0,
			"e":0,
			"ı":0,
			"i":0,
			"o":0,
			"ö":0,
			"u":0,
			"ü":0,
		}

		margin = len(y_data) // num_sections
		mults = []
		for n in range(num_sections):
			index = n * margin + margin // 2
			mults.append( y_data[index] )
		
		print(mults)
		print(predictions)

		for i in range(len(predictions)):
			for j in range(len(predictions[i])):
				mult = mults[i]
				res[vowels[j]] += mult * predictions[i][j]

		

		steps = []
		for pred in predictions:
			steps.append([vowels[np.argmax(pred)], float(max(pred))])
		
		print(steps)
		print(res)
		remove_all()

		response_wrapper = json.dumps({
			"array": res,
			"prediction": max(res, key=res.get),
			"steps": steps
		})
	except Exception as e:
		response_wrapper = json.dumps({
			"error": str(e)
		})
	finally:
		remove_all()
		return Response(content=response_wrapper, media_type="application/json")

@app.post("/analyze-parts")
async def anaylze_sound_parts(sound: UploadFile = File()):
	content_sound = await sound.read()

	rnd_id = str(uuid.uuid4())
	root_dir = "audios"
	path = os.path.join(root_dir, rnd_id + ".wav")

	with open(path, mode="wb") as f:
		f.write(content_sound)

	print(f'duration : {librosa.get_duration(filename=root_dir + "/" + rnd_id + ".wav")}')

	# create intervals
	audio, sr = librosa.load(f"audios/{rnd_id}.wav")

	print(f'sr : {sr}')
	print(f'len(audio) : {len(audio)}')
	buffer = int((0.03) * sr)
	samples_total = len(audio)
	samples_wrote = 0
	counter = 1
	padding = buffer // 3

	audios_names = []

	while samples_wrote < samples_total:

		#check if the buffer is not exceeding total samples 
		if buffer > (samples_total - samples_wrote):
			buffer = samples_total - samples_wrote

		block = audio[samples_wrote : (samples_wrote + buffer)]
		out_filename = "audios/split" + str(counter) + "_.wav"

		audios_names.append(out_filename)

		# Write 2 second segment
		sf.write(out_filename, block, sr)
		counter += 1
		samples_wrote += padding

		audio_to_spec(out_filename)

	predictions = predict_all(model_s)
	remove_all()

	x_data = np.arange(-5, 5, 0.001)
	y_data = stats.norm.pdf(x_data, 0, 1)
	n = len(predictions)

	vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]

	res = {
		"a":0,
		"e":0,
		"ı":0,
		"i":0,
		"o":0,
		"ö":0,
		"u":0,
		"ü":0,
	}

	margin = len(y_data) // len(predictions)

	for (i, pred) in enumerate(predictions):
		for (j, p) in enumerate(pred):
			mult = y_data[i * margin]
			res[vowels[j]] += mult * p
	
	print(predictions)
	print(res)

	return {"hello": "world"}



@app.post("/analyze")
async def anaylze_sound(sound: UploadFile = File()):
	start_time = time.time()
	try:
		content_sound = await sound.read()

		rnd_id = str(uuid.uuid4())
		root_dir = "audios"
		path = os.path.join(root_dir, rnd_id + ".wav")

		with open(path, mode="wb") as f:
			f.write(content_sound)

		print(f'duration : {librosa.get_duration(filename=root_dir + "/" + rnd_id + ".wav")}')

		audio_to_spec(path)
		prediction = predict_spec(model_s)
		response_wrapper = {
			"vowel" : prediction
		}
	except Exception as e:
		response_wrapper = {
			'msg':str(e)
		}
		pass
	finally:
		remove_all()
		return {"hello":"world"}
		# return Response(content=response_wrapper, media_type="application/json")

@app.post("/save_upload_file_tmp")
def save_upload_file_tmp(file: UploadFile) -> Path:
	root_dir = "audios"
	path = os.path.join(root_dir, file.filename)
	with open(path, "wb") as buffer:
		shutil.copyfileobj(file.file, buffer)

	audio_to_spec(path)
	
	return {"file_name":file.filename}

name = "barış"


# INITIAL LOADS
filename ="vowels_spec_model"
model_s = get_loaded_model_by_name(filename)
opt = SGD(lr=0.001)
model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'], run_eagerly=True)

