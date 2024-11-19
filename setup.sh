apt-get update && apt-get install wget

# tar -zxvf code.tar.gz
# tar -zxvf data.tar.gz

apt-get install tmux

cd code/
apt-get update; apt-get install build-essential ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt

pip install numba

pip install streamlit
pip install streamlit_shortcuts

pip install mlflow
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

apt install cron

apt install vim
export EDITOR=vim
service cron start

pip install -U albumentations


echo "Setup completed and Streamlit is running."
