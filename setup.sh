apt-get update && apt-get install wget

wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20240912160112/data.tar.gz
wget https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000315/data/20241024082938/code.tar.gz

tar -zxvf code.tar.gz
tar -zxvf data.tar.gz

apt-get install tmux

cd code/
apt-get update; apt-get install build-essential ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt

mv data code/

echo "Setup completed and Streamlit is running."
