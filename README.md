# five_distance_bbox



## The Detection Results and models of method ORCNN

**note**: The `ms` means multiple scale image split and the `rr` means random rotation.

### DOTA dataset


| Backbone | Lr schd | ms | rr | box AP |                           Baidu Yun                          |                                         Google Drive                                        |
|:--------:|:-------:|:--:|:--:|:------:|:------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|  R50-FPN |    1x   |  - |  - |  75.87 | [key: v4s0](https://pan.baidu.com/s/1OQtoPBi36zVwCk_jXmzbBw) | [model](https://drive.google.com/file/d/1Rv5sctUcnEDSGZhMxgXVZ-7dMai0qrIr/view?usp=sharing) |
| R101-FPN |    1x   |  - |  - |  76.28 | [key: zge9](https://pan.baidu.com/s/1M8VQo1CEhF-arFo2-Q_3uQ) | [model](https://drive.google.com/file/d/1Sz6CLjeCMAR06B1NfkbWnVX2FZuMCR8u/view?usp=sharing) |
|  R50-FPN |    1x   |  √ |  √ |  80.87 | [key: 66jf](https://pan.baidu.com/s/1d86ZqPQCSdoeXiQ38fvSyQ) | [model](https://drive.google.com/file/d/1tZjPOOioYtZKA3C1z6Twjcf__5NYvVOr/view?usp=sharing) |
| R101-FPN |    1x   |  √ |  √ |  80.52 | [key: o1r6](https://pan.baidu.com/s/1zUF4I09BjW8_pniy71cvtg) | [model](https://drive.google.com/file/d/1JG3V34PYiwZ3NM7KSLu9MRUPCO-RZOX5/view?usp=sharing) |

### HRSC2016 dataset

| Backbone | Lr schd | ms | rr | voc07 | voc12 |                           Baidu Yun                          |                                         Google Drive                                        |
|:--------:|:-------:|:--:|:--:|:-----:|:-----:|:------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|
|  R50-FPN |    3x   |  - |  - |  90.4 |  96.5 | [key: 02zc](https://pan.baidu.com/s/1ISxj1HTumqhD-tjwMYcRhg) | [model](https://drive.google.com/file/d/1K_RvwPMtDl_amR_lMxeiXzVSwNFchPHc/view?usp=sharing) |
| R101-FPN |    3x   |  - |  - |  90.5 |  97.5 | [key: q3e6](https://pan.baidu.com/s/19x1doXr2qqy7OOTAMKzazA) | [model](https://drive.google.com/file/d/1SZhO4HWzstjbzI3SEwGun2byM3y4p9Bc/view?usp=sharing) |

## Citation


**note**: If you have questions or good suggestions, feel free to propose issues and contact me.

### Major features

- **MMdetection feature inheritance**

  OBBDetection doesn't change the structure and codes of original MMdetection and the additive codes are under MMdetection logic. Therefore, our OBBDetection inherits all features from MMdetection.

- **Support of multiple frameworks out of box**

  We implement multiple oriented object detectors in this toolbox (*e.g.* RoI Transformer, Gliding Vertex). Attributing to moudlar design of MMdetection, Many parts of detectors (*e.g.* backbone, RPN, sampler and assigner) have multiple options.

- **Flexible representation of oriented boxes**

  Horizontal bounding boxes (HBB), oriented bounding boxes (OBB) and 4 point boxes (POLY) are supported in this toolbox. The program will confirm the type of bounding box by the tensor shape or the default setting.

We develop [BboxToolkit](https://github.com/jbwang1997/BboxToolkit) to support oriented bounding boxes operations, which is heavily depended on by this toolbox.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Update

- (**2021-09-18**) Implement [Double Head OBB](configs/obb/double_heads_obb) in the OBBDetection.
- (**2021-09-01**) Implement [FCOS OBB](configs/obb/fcos_obb) in the OBBDetection.
- (**2021-08-21**) Reimplement the [PolyIoULoss](configs/obb/poly_iou_loss).


@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
