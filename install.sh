sudo apt install build-essential
! wget https://raw.githubusercontent.com/ifzhang/ByteTrack/main/README.md
pip3 install -r requirements.txt
python3 setup.py  develop
pip3 install cython
pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
pip install onemetric

pip install -q loguru lap thop
