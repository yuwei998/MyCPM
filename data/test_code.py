from DataSet import *
from utils import *
import cv2
import numpy as np
data_dir='.'
batch_size=5
input_size=368
hm_size=46
normalize=True
category='blouse'
gaussian_variance=1
joints_num= 13
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]


step=50
data=DataSet(data_dir, batch_size, input_size, hm_size, normalize,category, gaussian_variance,joints_num ,sample_set='train')
data_generator=data.data_generator
for i in range(step):
    img,hm=next(data_generator)
    img=(img[0]+0.5)*256
    hm=np.expand_dims(hm,axis=0)
    
    img_save=visualize_result(img, hm, joints_num, hm_size, joint_color_code)
    
    cv2.imwrite('./result/hm'+ str(i)+ '.jpg',img_save)