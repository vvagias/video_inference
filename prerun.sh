wget https://bootstrap.pypa.io/get-pip.py
sudo chmod +x get-pip.py
python get-pip.py -y
pip install jupyterlab
apt update
apt install ffmpeg libsm6 libxext6  -y