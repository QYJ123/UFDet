from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
from PIL import Image
import matplotlib.pyplot as plt
import os,gc

#hrsc dataset
image_file_path = r'./data/hrsc/Test/AllImages/' #需要加载的测试图片的文件路径

#diffferent methods

save_file_path = r'./test_img/save_imgs/ufdet/hrsc_101test/' #保存测试图片的路径
config_file = r'./configs/obb/my_improve3/1-ORCNN_pr_r101_hrsc.py' #网络模型
checkpoint_file = r'./work_dirs/1-ORCNN_pr_r101_hrsc0/epoch_35.pth'  #训练好的模型参数


os.makedirs(save_file_path,exist_ok=True)
# # data dataset
# imagepath = r'./data/DOTA1_0/train/images/' #train test val 需要加载的测试图片的文件路径
# savepath = r'./test_img/save_imgs/voc/' #保存测试图片的路径
# config_file = r'./configs/1_my_model/faster_rcnn_r50_fpn_1x_coco.py' #网络模型
# checkpoint_file = r'./work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_9.pth'  #训练好的模型参数
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image

if __name__ == '__main__': 
    image_list_sort  = os.listdir(image_file_path)
    image_list_sort.sort()
    #print(len(image_list_sort))
    for i in range(0,len(image_list_sort),1):
        filename = image_list_sort[i]
        img_path = os.path.join(image_file_path, filename)
        result = inference_detector(model, img_path)
        img1 = show_result_pyplot(model, img_path, result,score_thr=0.6)
        img = Image.fromarray(img1)
        filename1 = filename.split('.')[0] + '.jpg'
        out_file = os.path.join(save_file_path, filename1)
        img.save(out_file)
        print(i,'finish save %s'%filename1)
        del  filename,result,img1,img
        gc.collect()
    print('finish test all images!')

'''filename = '100000003.bmp'
img = os.path.join(image_file_path, filename)
result = inference_detector(model, img)
img = show_result_pyplot(model, img, result,score_thr=0.6)
plt.imshow(img)
plt.show()
img = Image.fromarray(img)
filename1 = filename.split('.')[0] + '.jpg'
out_file = os.path.join(save_file_path, filename1)


img.save(out_file)
print('finish save %s'%filename1)'''





