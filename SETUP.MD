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

## Prepare the data (run once)

Precondition:

```bash
du -h --max-depth 1 VisDrone/ | sort -h
```

```txt
81M     VisDrone/VisDrone2019-DET-val
296M    VisDrone/VisDrone2019-DET-test-challenge
306M    VisDrone/VisDrone2019-DET-test-dev
1.5G    VisDrone/VisDrone2019-DET-train
2.2G    VisDrone/
```

Prepare:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  # optional
mv sam_vit_h_4b8939.pth weights/  # optional
python scripts/data_prepare.py --dataset VisDrone
```

Using the downloaded weight might cause CUDA OOM (even with 24GB).
If you skip the checkpoint (don't run `wget` and `mv`), tradeoff quality for speed then:

```txt
scripts/data_prepare.py:32: UserWarning: It is recommended to install segment-anything for better pseudo masks. See instructions in README.md.
100%|████████| 1610/1610 [01:13<00:00, 21.80it/s]  # test-dev
100%|████████| 6471/6471 [06:51<00:00, 15.73it/s]  # train
100%|████████| 548/548 [00:23<00:00, 22.92it/s]    # val
```

Post-condition:

```bash
du -h --max-depth 1 VisDrone/ | sort -h
```

```txt
724K    VisDrone/split
296M    VisDrone/VisDrone2019-DET-test-challenge
2.2G    VisDrone/VisDrone2019-DET-val
8.1G    VisDrone/VisDrone2019-DET-test-dev
41G     VisDrone/VisDrone2019-DET-train
52G     VisDrone/
```

- Purpose of this step: Convert VisDrone annotations into YOLO labels and generate per-pixel training targets. Regions labeled `0` (ignored) or `11` (others) are gray-masked; valid classes `1`-`10` are written as model classes `0`-`9`.
- Output of this step: Update each split with the prepared image paths and create labels, masks, and split files used during training.

    | Where | Description | 
    | -- | -- |
    | `VisDrone/VisDrone-.../images` | Original images, plus `_masked.jpg` copies when ignored/other regions are present |
    | `VisDrone/VisDrone-.../labels` | YOLO-compatible `.txt` files containing valid objects only |
    | `VisDrone/VisDrone-.../masks` | `.npy` files of shape `(H, W, 2)`: object heatmap and loss-weight map (smaller objects get more weight) |
    | `VisDrone/split` | `train.txt`, `val.txt`, `test-dev.txt`, and `test-challenge.txt` listing the images used by each split |

## Main programs

```bash
# run detect.py
cp data/images/visdrone.txt data/images/sanity.txt
echo "./VisDrone/VisDrone2019-DET-val/images/0000277_02001_d_0000549.jpg" > data/images/sanity.txt 
python detect.py --weights weights/esod_yolov5m.pt --source data/images/sanity.txt --img-size 1536 --device 0 --view-cluster --line-thickness 1
```