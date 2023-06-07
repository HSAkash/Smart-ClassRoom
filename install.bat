pip install -r requirements.txt
python setup.py  develop
pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install cython_bbox
pip install onemetric
pip install -q loguru lap thop