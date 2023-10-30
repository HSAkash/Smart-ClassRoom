# SmartClassRoom
## Features
* Automatic attendance
* Student Activity detection
* Student Activity Tracking

[](/assets/smartClassroom.mp4)

## Installation

### Conda
```
conda create --name SmartClassRoom python=3.10
```
### You can activate it with the following commands.
```
conda activate SmartClassRoom
```

### Window
```
install.bat
```
### Linux
```
bash install.sh
```








### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Otherwise,
```
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
```

Step3. Others
```shell
pip3 install cython_bbox
```

