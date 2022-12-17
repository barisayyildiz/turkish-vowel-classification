import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

import time

import os

from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import img_to_array,load_img,array_to_img

from keras.preprocessing.image import DirectoryIterator,DataFrameIterator

def audio_to_spec(path: str):
	file_name = path.split("/")[-1]

	y, sr = librosa.load(path)
	S = librosa.feature.melspectrogram(y=y, sr=sr)

	cmap = plt.get_cmap('inferno')
	plt.figure(figsize=(8,8))

	plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
	plt.savefig(f'specs/images/{file_name.split(".")[0]}.png')


def get_loaded_model_by_name(filename):
  # load json and create model
  json_file = open(f'models/{filename}.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(f'models/{filename}.h5')
  print("Loaded model from disk")
  # print(loaded_model.summary())
  return loaded_model


def load_data_from_generators(generator,timestep,color_mode ='grayscale',target_size =(96,96)):
	data_frames = []
	dat_labels = []

	for i in range(0,len(generator.filepaths),timestep):
		time_dist_data = []
		for k in range(timestep):
			time_dist_data.append(img_to_array(load_img(generator.filepaths[i+k],color_mode=color_mode,target_size=target_size)))
		dat_labels.append(generator.labels[i])
		data_frames.append(time_dist_data)

	return np.array(data_frames),np.array(dat_labels)


def predict_spec(model_s):
	h = 96
	w = 96

	path_spec_test = "specs"

	spec_datagen_test = ImageDataGenerator()
	test_generator_spec = spec_datagen_test.flow_from_directory(path_spec_test,class_mode='categorical',color_mode ='rgb',shuffle = False,  target_size =(h,w),subset='training')
	test_spec_data,test_labels_spec = load_data_from_generators(test_generator_spec,timestep=1,color_mode='rgb')

	vowels = ["a", "e", "ı", "i", "o", "ö", "u", "ü"]
	prediction = model_s.predict(test_spec_data)[0]

	return vowels[np.argmax(prediction)]



def remove_file():
	for f in os.listdir('audios'):
		os.remove(os.path.join('audios', f))
	for f in os.listdir('specs/images'):
		os.remove(os.path.join('specs/images', f))
