import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def audio_to_spec(path: str):
	file_name = path.split("/")[-1]

	y, sr = librosa.load(path)
	S = librosa.feature.melspectrogram(y=y, sr=sr)

	cmap = plt.get_cmap('inferno')
	plt.figure(figsize=(8,8))

	plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
	plt.savefig(f'specs/{file_name}.png')
