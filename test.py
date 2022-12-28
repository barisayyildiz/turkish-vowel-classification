import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import img_to_array,load_img,array_to_img

from keras.preprocessing.image import DirectoryIterator,DataFrameIterator
import numpy as np

import soundfile as sf

from scipy import stats

from utils import audio_to_spec, predict_all, remove_all, get_loaded_model_by_name

def create_spectograms(filename):
	# audio, sr = librosa.load(filename)
	wave, sr = librosa.load(filename, sr=None)

	# print(f'sr : {sr}')
	# print(f'len(audio) : {len(audio)}')
	# buffer = int((0.03) * sr)
	# samples_total = len(audio)
	# samples_wrote = 0
	# counter = 1
	# padding = buffer

	# audios_names = []

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
		mults.append(
			y_data[index]
		)

	# margin = len(y_data) // len(num_sections)
	# mults = []
	# print(len(y_data))
	# print(np.split(y_data, num_sections))
	# for y in np.split(y_data, num_sections):
	# 	print(len(y), len(y)//2)
	# 	mults.append(
	# 		y[len(y)//2]
	# 	)
	# # mults = [y[len(y)//2] for y in np.split(y_data, num_sections)]
	print(mults)
	print(predictions)

	for i in range(len(predictions)):
		for j in range(len(predictions[i])):
			mult = mults[i]
			res[vowels[j]] += mult * predictions[i][j]
	
	print(res)
	remove_all()

	# for (i, pred) in enumerate(predictions):
	# 	for (j, p) in enumerate(pred):
	# 		mult = y_data[(i+1) * margin]
	# 		res[vowels[j]] += mult * p
	
	# print(predictions)
	# print(res)
	# remove_all()


	# while samples_wrote < samples_total:

	# 	#check if the buffer is not exceeding total samples 
	# 	if buffer > (samples_total - samples_wrote):
	# 		buffer = samples_total - samples_wrote

	# 	block = audio[samples_wrote : (samples_wrote + buffer)]
	# 	out_filename = "audios/split" + str(counter) + "_.wav"

	# 	audios_names.append(out_filename)

	# 	# Write 2 second segment
	# 	sf.write(out_filename, block, sr)
	# 	counter += 1
	# 	samples_wrote += padding

	# 	audio_to_spec(out_filename)

	# predictions = predict_all(model_s)

	# vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]

	# x_data = np.arange(-5, 5, 0.001)
	# y_data = stats.norm.pdf(x_data, 0, 1)
	# n = len(predictions)

	# res = {
	# 	"a":0,
	# 	"e":0,
	# 	"ı":0,
	# 	"i":0,
	# 	"o":0,
	# 	"ö":0,
	# 	"u":0,
	# 	"ü":0,
	# }

	# margin = len(y_data) // len(predictions)

	# for (i, pred) in enumerate(predictions):
	# 	for (j, p) in enumerate(pred):
	# 		mult = y_data[(i+1) * margin]
	# 		res[vowels[j]] += mult * p
	
	# print(predictions)
	# print(res)
	# remove_all()



# INITIAL LOADS
filename ="vowels_spec_model"
model_s = get_loaded_model_by_name(filename)
opt = SGD(lr=0.001)
model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'], run_eagerly=True)




create_spectograms("tests/sök.wav")
