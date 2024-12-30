import BboxToolkit as bt

import copy
import mmcv
import numpy as np

from mmdet.core import eval_arb_map, eval_arb_recalls
from ..builder import DATASETS
from ..custom import CustomDataset

from mmcv import print_log
#from collection import OrderedDict

@DATASETS.register_module()
class DIORDataset(CustomDataset):

    CLASSES = bt.get_classes('dior')

    def __init__(self,
                 xmltype,
                 imgset,
                 ann_file,
                 img_prefix,
                 *args,
                 **kwargs):
        assert xmltype in ['hbb', 'obb']
        self.xmltype = xmltype
        self.imgset = imgset
        super(DIORDataset, self).__init__(*args,
                                          ann_file=ann_file,
                                          img_prefix=img_prefix,
                                          **kwargs)

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            cls.custom_classes = False
            return cls.CLASSES

        cls.custom_classes = True
        return bt.get_classes(classes)

    def load_annotations(self, ann_file):
        contents, _ = bt.load_dior(
            img_dir=self.img_prefix,
            ann_dir=ann_file,
            classes=self.CLASSES,
            xmltype=self.xmltype)
        if self.imgset is not None:
            contents = bt.split_imgset(contents, self.imgset)
        return contents

    def pre_pipeline(self, results):
        results['cls'] = self.CLASSES
        super().pre_pipeline(results)

    def format_results(self, results, save_dir=None, **kwargs):
        assert len(results) == len(self.data_infos)
        contents = []
        for result, data_info in zip(results, self.data_infos):
            info = copy.deepcopy(data_info)
            info.pop('ann')

            ann, bboxes, labels, scores = dict(), list(), list(), list()
            for i, dets in enumerate(result):
                bboxes.append(dets[:, :-1])
                scores.append(dets[:, -1])
                labels.append(np.zeros((dets.shape[0], ), dtype=np.int) + i)
            ann['bboxes'] = np.concatenate(bboxes, axis=0)
            ann['labels'] = np.concatenate(labels, axis=0)
            ann['scores'] = np.concatenate(scores, axis=0)
            info['ann'] = ann
            contents.append(info)

        if save_dir is not None:
            bt.save_pkl(save_dir, contents, self.CLASSES)
        return contents

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 iou_thr=0.5,#np.linspace(0.5, 0.95, 10),
                 scale_ranges=None,
                 use_07_metric=True,
                 proposal_nums=(100, 300, 1000)):
        #np.linspace(0.5, 0.95, 10)[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        mean_mAP50_95 = []
        if metric == 'mAP':
            if isinstance(iou_thr, float):
                mean_ap, _ = eval_arb_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger)
                eval_results['mAP'] = mean_ap
            else:
                #eval_results = OrderedDict()
                mean_aps = []
                
                for iou in iou_thr:
                    print_log(f'\n{"-" * 15}iou_thr: {iou}{"-" * 15}')
                    mean_ap, _ = eval_arb_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger)
                    mean_aps.append(mean_ap)
                    eval_results[f'AP{int(iou * 100):02d}'] = round(mean_ap, 3)
                    eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
                    mean_mAP50_95.append(round(mean_ap,4))
                print(mean_mAP50_95)
                print('mAP75:{}'.format(mean_mAP50_95[5]))
                print('mAP50:95:{}'.format(sum(mean_mAP50_95) / len(mean_mAP50_95)) )
                #eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_arb_recalls(
                gt_bboxes, results, True, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
