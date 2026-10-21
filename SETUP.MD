# Environment setup

```bash
# install dependencies
conda create --prefix /mnt/wd-storage/thainguyen/env/esod python=3.8.20
conda activate /mnt/wd-storage/thainguyen/env/esod
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0
pip install -r requirements.txt
cd third_party/segment-anything/
pip install -e .
cd -

# download (from https://github.com/VisDrone/VisDrone-Dataset) and organize data
mkdir -p data/VisDrone/VisDrone2019-DET-{train,val,test-dev,test-challenge}
# then create a soft link
ln -sf data/VisDrone VisDrone

# manually download weights from https://drive.google.com/drive/folders/1jrrtG34q6gqdGMx7SflNK1uM6iiVKpin or the following
mkdir weights
cd weights
pip install gdown
gdown 1vDHHCw33vqYFpJvcNJ32G4IUQIrKvimX
gdown 1ZivdkER0KfVZnKI7mcjwgnDqSiIQZJ22
cd -
```

# Running scripts

```bash
# run detect.py
cp data/images/visdrone.txt data/images/sanity.txt
echo "./VisDrone/VisDrone2019-DET-val/images/0000277_02001_d_0000549.jpg" > data/images/sanity.txt 
python detect.py --weights weights/esod_yolov5m.pt --source data/images/sanity.txt --img-size 1536 --device 0 --view-cluster --line-thickness 1
```