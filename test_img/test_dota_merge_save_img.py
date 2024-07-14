from mmdet.apis import init_detector, inference_detector
from mmdet.apis import show_result_pyplot
from PIL import Image
#import matplotlib.pyplot as plt
import os,time,gc,argparse
from mmdet.apis import inference_detector_huge_image

#dota dataset
image_file_path = r'/home/yajun/CX/Pycharm/mmdet_add/my_obb/data/DOTA1_0/test/images/' #需要加载的测试图片的文件路径

save_file_path = r'./test_img/save_imgs/qpdet/dota50_sstestmerge/' #保存测试图片的路径
config_file = r'./configs/qpdet_r50_fpn_1x_ss_dota.py' #网络模型
checkpoint_file = r'./work_dirs/qpdet_r50_fpn_1x_ss_dota/epoch_12.pth'  #训练好的模型参数

split = './BboxToolkit/tools/split_configs/dota1_0/ss_test.json'

os.makedirs(save_file_path,exist_ok=True)
device = 'cuda:0'


# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
nms_cfg = dict(type='BT_nms', iou_thr=0.5)

def parse_args():
    parser = argparse.ArgumentParser(description = 'input int_1 and int_2')
    parser.add_argument('start_a', help = 'start int1')
    parser.add_argument('end_b', help = 'end int2')
    args = parser.parse_args()
    return args
def main():
    arg = parse_args()
    image_list_sort  = os.listdir(image_file_path)
    image_list_sort.sort()
    #print(len(image_list_sort))
    for i in range(int(arg.start_a),int(arg.end_b)):#each 0,470,len(image_list_sort)
            filename = image_list_sort[i]
            result = inference_detector_huge_image(model,os.path.join(image_file_path, filename), split, nms_cfg)
            img1 = show_result_pyplot(model, os.path.join(image_file_path, filename), result,score_thr=0.6)
            img = Image.fromarray(img1)
            img.save(os.path.join(save_file_path, filename.split('.')[0] + '.jpg'))
            print(i,'finish save %s'% (filename.split('.')[0] + '.jpg'))
            del  filename,result,img1,img
            gc.collect()
    print('finish test all images!')

if __name__ == '__main__':
    main()
    

