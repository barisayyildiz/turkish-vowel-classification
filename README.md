# grad-project-backend

### [Deep Learning Model](https://colab.research.google.com/drive/1tspPmK4ZZWOSvyUBaS6asgnuHoOznA-a)

### setup
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

