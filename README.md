# [JSAC 2022] VaBUS: Edge-Cloud Real-time Video Analytics via Background Understanding and Subtraction

VaBUS is an edge-cloud video analytics system by leveraging background understanding and subtraction. This repo is the implementation of [VaBUS](https://ieeexplore.ieee.org/document/9953098), JSAC 2022.

```bibtex
@ARTICLE{wang2022vabus,
    author={Wang, Hanling and Li, Qing and Sun, Heyang and Chen, Zuozhou and Hao, Yingqian and Peng, Junkun and Yuan, Zhenhui and Fu, Junsheng and Jiang, Yong},
    journal={IEEE Journal on Selected Areas in Communications},
    title={VaBUS: Edge-Cloud Real-time Video Analytics via Background Understanding and Subtraction},
    year={2022},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/JSAC.2022.3221995}}
```

---------------------------

## Environment
Cloud: Python3.8.10 and Ubuntu20.04

Edge: Python3.6.9, C++, and Ubuntu18.04 (Jetson Xavier NX)

## Requirement
1. For Python, see *src/requirements.txt*.
2. For Jetson RoI encoding, you need to rebuild ffmpeg. Please refer to *jetson_roi_encode/*

## Modules and file structures
- *dataset/*: put your dataset (video or image folder) here
- *dataset/ground_truth/*: ground_truth results generated by *src/generate_encode_size.py* and *src/generate_inference_res.py* for evaluation purpose
- *jetson_roi_encode/*: for RoI encoding on Jetson
- *results/*: evaluation results (auto-generated)
- *src/cloud.py*: main script for cloud running
- *src/edge.py*: main script for edge running
- *src/param.yml*: parameter files used by *src/cloud.yml* and *src/edge.yml*
- others should be self-explainable

## Get started

## 1. On the cloud server
run `python cloud.py`

## 2. On the edge device
run `python edge.py`

## Notice
- Build the ffmpeg library for video encoding on Jetson
- Carefully set up the model files under *src/models* to adapt to your machine
- We cleaned up the code for clarity.
- Any question please ask. Thanks!
