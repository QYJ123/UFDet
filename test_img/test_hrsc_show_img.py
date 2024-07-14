from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
from PIL import Image
import matplotlib.pyplot as plt
import os

#hrsc dataset
imagepath = r'./data/hrsc/Test/AllImages/' #需要加载的测试图片的文件路径
#savepath = r'./test_img/save_imgs/hrsc_101/' #保存测试图片的路径

#os.makedirs(savepath,exist_ok=True)

###hrsc orcnn##
#config_file = r'./configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_3x_hrsc.py' #网络模型
#checkpoint_file = r'./work_dirs/faster_rcnn_orpn_r50_fpn_3x_hrsc/epoch_36.pth'  #训练好的模型参数
###hrsc s2anet##
config_file = r'./configs/obb/s2anet/s2anet_r50_fpn_3x_hrsc.py' #网络模型
checkpoint_file = r'./work_dirs/s2anet_r50_fpn_3x_hrsc/epoch_36.pth'  #训练好的模型参数



# data dataset
# imagepath = r'./data/DOTA1_0/train/images/' #train test val 需要加载的测试图片的文件路径
# savepath = r'./test_img/save_imgs/voc/' #保存测试图片的路径
# config_file = r'./configs/1_my_model/faster_rcnn_r50_fpn_1x_coco.py' #网络模型
# checkpoint_file = r'./work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_9.pth'  #训练好的模型参数
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
 
filename = '100000005.bmp'
img = os.path.join(imagepath, filename)
result = inference_detector(model, img)
img = show_result_pyplot(model, img, result,score_thr=0.6)
plt.imshow(img)
plt.show()
#img = Image.fromarray(img)
#filename1 = filename.split('.')[0] + '.jpg'
#out_file = os.path.join(savepath, filename1)
#img.show()

#img.save(out_file)
#print('finish save %s'%filename1)
