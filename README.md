# Turkish Vowel Classification using Deep Learning

## Description
This project utilizes deep learning techniques to classify eight different vowels in the Turkish language and aims to address speech disorders, particularly among children. It includes the development of a deep learning model trained on a dataset of speech samples and the creation of a user-friendly web application for real-time vowel sound classification.

The model utilizes computer vision techniques such as Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to accurately classify Turkish vowel sounds. The training dataset consists of audio recordings of participants pronouncing three-letter words with vowels in the middle, along with ultrasound recordings to capture tongue movements.

The data is preprocessed by combining audio and video files and annotating vowels using Praat software. Spectrograms are generated from the audio data and used as input for the model. The web application, built using vanilla JavaScript, features an International Phonetic Alphabet (IPA) chart and a button to control microphone input. The system automatically starts recording when the sound level exceeds a threshold and stops when it drops below. The recorded audio is converted to .wav format and sent to the backend service for vowel prediction.

The backend, developed using Python and FastAPI, provides a single API that accepts the user's .wav file and converts it into spectrograms. These spectrograms are then fed into the trained deep learning model, which predicts the corresponding vowel sound. The model divides the sound file into 30ms segments and adjusts predictions based on a Gaussian distribution. The prediction with the highest resulting value is returned.

The project is deployed on a Google Cloud Platform (GCP) virtual machine, with the frontend and backend code deployed together for improved response times. The frontend code is bundled using Webpack, and the backend server is started using Nginx.

## Key Features
- Deep learning model for accurately classifying eight distinct vowel sounds in the Turkish language.
- User-friendly web application with an IPA chart and automatic audio recording.
- Frontend implemented using vanilla JavaScript for seamless integration with external libraries.
- Backend service developed using Python and the FastAPI library for efficient handling of requests.
- Spectrograms used as input to the model for vowel sound classification.
- Deployed on a GCP virtual machine for reliable and responsive performance.

## Usage
- Open the web application in your browser.
- Allow access to your microphone when prompted.
- Pronounce a Turkish vowel sound into the microphone.
- The application will automatically start recording when the sound level exceeds the threshold.
- The recording will stop when the sound level drops below the threshold.
- The recorded sound will be converted to a .wav file and sent to the backend for vowel prediction.
- The predicted vowel will be displayed on the IPA chart.


## [Deep Learning Model](https://colab.research.google.com/drive/1tspPmK4ZZWOSvyUBaS6asgnuHoOznA-a)

## Installation
```
git clone https://github.com/barisayyildiz/grad-project-backend.git
cd grad-project-backend
sudo pip install virtualenv
virtualenv venv
source ./venv/bin/activate
mkdir audios
mkdir specs
cd specs
mkdir images
cd ..
pip install -r requirements.txt
uvicorn main:app --reload
```


### setup vm & nginx
```
sudo apt-get update
sudo apt install -y python3-pip nginx
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
# paste nginx config file
sudo service nginx restart
```

#### nginx config file
```
server {
  listen 80;
  server_name SERVER_NAME;
  location / {
    proxy_pass http://127.0.0.1:8000;
   }
}
```

