import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

chunk_name = "test_sound.wav"
y, sr = librosa.load(chunk_name)
S = librosa.feature.melspectrogram(y=y, sr=sr)

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))

plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
plt.savefig(f'specs/spectogram.png')
plt.clf()



def get_loaded_model_by_name(filename):
	# load json and create model
	json_file = open(f'model/{filename}.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(f'model/{filename}.h5')
	print("Loaded model from disk")
	# print(loaded_model.summary())
	return loaded_model


# filename ="vowels_spec_model"
# model_s = get_loaded_model_by_name(filename)
# model_s.predict(S)

# opt = SGD(lr=0.001)
# model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'])






