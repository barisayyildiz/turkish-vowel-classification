import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import model_from_json

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import img_to_array,load_img,array_to_img

from keras.preprocessing.image import DirectoryIterator,DataFrameIterator
import numpy as np


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


print("Evaluate on spectrogram test data")
filename ="vowels_spec_model"
model_s = get_loaded_model_by_name(filename)
opt = SGD(lr=0.001)
model_s.compile(loss = "categorical_crossentropy", optimizer =opt,metrics=['accuracy'], run_eagerly=True)

################ Spectrogram images ##############


spec_datagen = ImageDataGenerator()

h = 96
w = 96

# path_spec_train = "/content/train_vowels_/annotated_specs"
path_spec_test = "specs"



# train_generator_spec = spec_datagen.flow_from_directory(path_spec_train,class_mode='categorical', color_mode ='rgb', shuffle = False, target_size =(h,w),subset='training')

spec_datagen_test = ImageDataGenerator()

test_generator_spec = spec_datagen_test.flow_from_directory(path_spec_test,class_mode='categorical',color_mode ='rgb',shuffle = False,  target_size =(h,w),subset='training')


# train_spec_data,train_labels_spec = load_data_from_generators(train_generator_spec,timestep=1,color_mode='rgb')
test_spec_data,test_labels_spec = load_data_from_generators(test_generator_spec,timestep=1,color_mode='rgb')

print(test_spec_data.shape)


print(model_s.predict(test_spec_data))


# # be careful if you run the training, do not run these lines
# # train_spec_data_reshape  = train_spec_data.reshape( train_spec_data.shape[0], 1, train_spec_data.shape[1],train_spec_data.shape[2], train_spec_data.shape[3] )
# # test_spec_data_reshape  = test_spec_data.reshape( test_spec_data.shape[0], 1, test_spec_data.shape[1],test_spec_data.shape[2], test_spec_data.shape[3] )
# #


# results = model_s.evaluate(test_spec_data_reshape,test_labels_spec, batch_size=10)
# print("test loss, test acc:", results)

# # for only spec data
# import seaborn as sns
# from sklearn.metrics import confusion_matrix

# predictions = model_s.predict(test_spec_data_reshape)

# y_true = np.argmax(test_labels_spec,axis=1)
# y_pred = np.argmax(predictions,axis=1)



