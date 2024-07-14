from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
from PIL import Image
import matplotlib.pyplot as plt
import os
from mmdet.apis import inference_detector_huge_image

#hrsc dataset
imagepath = r'/home/yajun/CX/Pycharm/mmdet_add/my_obb/data/DOTA1_0/test/images/' #需要加载的测试图片的文件路径
#savepath = r'./test_img/save_imgs/DOTA/DOTA_Pre/' #保存测试图片的路径
#os.makedirs(savepath,exist_ok=True)
save_file_path = r'./test_img/save_imgs/1-dh_faster/dota_test/' #保存测试图片的路径
config_file = r'./configs/obb/double_heads_obb/dh_faster_rcnn_obb_r50_fpn_1x_dota10.py' #网络模型
checkpoint_file = r'./work_dirs/dh_faster_rcnn_obb_r50_fpn_1x_dota10/epoch_12.pth'  #训练好的模型参数

device = 'cuda:0'
split = './BboxToolkit/tools/split_configs/dota1_0/ms_test.json'
# init a detector
# inference the demo image
model = init_detector(config_file, checkpoint_file, device=device) 
filename = 'P1485.png'
img = os.path.join(imagepath, filename)
nms_cfg = dict(type='BT_nms', iou_thr=0.5)
result = inference_detector_huge_image(model, img, split, nms_cfg)
# show the results
img = show_result_pyplot(model, img, result,score_thr=0.6)
plt.imshow(img)
plt.show()
#img = Image.fromarray(img)
#filename1 = filename.split('.')[0] + '.jpg'
#out_file = os.path.join(savepath, filename1)
#img.show()

#img.save(out_file)
#print('finish save %s'%filename1)
