# five_distance_bbox
# RPGAOD

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
