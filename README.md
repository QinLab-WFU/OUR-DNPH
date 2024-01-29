# Deep Neighborhood-aware Proxy Hashing with Uniform Distribution Constraint for Cross-modal Retrieval [Paper](https://dl.acm.org/doi/10.1145/3643639)
This paper is accepted for publication with TOMM.

## Dependencies

- pytorch 1.12.1
- sklearn
- tqdm
- pillow

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Train

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --num-classes 80


### Citation
@article{10.1145/3643639,  
author = {Huo, Yadong and Qin, Qibing and Dai, Jiangyan and Zhang, Wenfeng and Huang, Lei and Wang, Chengduan},  
title = {Deep Neighborhood-aware Proxy Hashing with Uniform Distribution Constraint for Cross-modal Retrieval},  
year = {2024},  
journal = {ACM Transactions on Multimedia Computing, Communications, and Applications},  
doi = {10.1145/3643639}}


### Acknowledgements
[DCHMT](https://github.com/kalenforn/DCHMT)
