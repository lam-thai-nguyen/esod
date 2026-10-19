```bash
conda create --prefix /mnt/wd-storage/Thai/env/esod python=3.8.20
conda activate /mnt/wd-storage/Thai/env/esod
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install setuptools==59.5.0
pip install -r requirements.txt
cd third_party/segment-anything/
pip install -e .
cd -

# download VisDrone2019-DET-val.zip to data/VisDrone
# and create a soft link
mkdir data/VisDrone/VisDrone2019-DET-val
unzip data/VisDrone/VisDrone2019-DET-val.zip -d data/VisDrone/VisDrone2019-DET-val
ln -sf data/VisDrone VisDrone
rm data/VisDrone/VisDrone2019-DET-val.zip

# get weights from https://drive.google.com/drive/folders/1jrrtG34q6gqdGMx7SflNK1uM6iiVKpin
# put it under weights/
mv ./esod_yolov5m.pt ./weights/

# run detect.py (sanity check)
cp data/images/visdrone.txt data/images/sanity.txt
echo "./VisDrone/VisDrone2019-DET-val/images/0000277_02001_d_0000549.jpg" > data/images/sanity.txt 
python detect.py --weights weights/esod_yolov5m.pt --source data/images/sanity.txt --img-size 1536 --device 0 --view-cluster --line-thickness 1
```