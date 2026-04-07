# GALACLIP

GALACLIP is a CLIP-based semantic segmentation codebase built on MMSegmentation.
This repository provides the core training and evaluation code under `configs/`, `models/`, `train.py`, and `test.py`.

## Environment

Install dependencies with a PyTorch + MMCV/MMSeg stack compatible with this repo:

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.4.4 mmsegmentation==0.24.0
pip install scipy timm==0.3.2
```

## Data and Pretrained Weights

1. Prepare datasets following MMSegmentation dataset conventions.
2. Download CLIP pretrained weights (for example `ViT-B-16.pt`) and update the `pretrained` path in your config.

## Training

VOC12 inductive:

```bash
bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py output/voc_20/inductive
```

VOC12 transductive:

```bash
bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_st.py output/voc_20/transductive --load-from=output/voc_20/inductive/iter_20000.pth
```

COCO inductive:

```bash
bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py output/coco_156/inductive
```

COCO transductive:

```bash
bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_40k_12_100_multi_st.py output/coco_156/transductive --load-from=output/coco_156/inductive/iter_80000.pth
```

Fully supervised:

```bash
bash dist_train.sh configs/voc12/vpt_seg_fully_vit-b_512x512_20k_12_10.py output/voc_20/fully
bash dist_train.sh configs/coco/vpt_seg_fully_vit-b_512x512_80k_12_100_multi.py output/coco_171/fully
```

## Evaluation

```bash
python test.py <config_path> <checkpoint_path> --eval=mIoU
```

Examples:

```bash
python test.py configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py output/voc_20/inductive/latest.pth --eval=mIoU
python test.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py output/coco_156/inductive/latest.pth --eval=mIoU
python test.py configs/cross_dataset/coco-to-voc.py output/coco_156/inductive/iter_80000.pth --eval=mIoU
python test.py configs/cross_dataset/coco-to-context.py output/coco_156/inductive/iter_80000.pth --eval=mIoU
```

## Acknowledgement

This repository is built on top of open-source CLIP, MMSegmentation, and prior zero-shot semantic segmentation implementations.
