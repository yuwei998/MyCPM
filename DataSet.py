import numpy as np
import cv2
from math import ceil
import random
from PIL import Image
from PIL import ImageEnhance 

class DataSet():
    def __init__(self, data_dir, batch_size, input_size, hm_size, normalize,category,gaussian_variance, joints_num, center_radius, sample_set='train'):
        self.train_set_img, self.train_set_joints, self.train_set_bx,\
        self.valid_set_img, self.valid_set_joints, self.valid_set_bx = self.read_file(data_dir,category)
        self.data_generator=self.generator(data_dir,batch_size, input_size,hm_size, normalize, category,gaussian_variance ,joints_num, center_radius,sample_set)
    def generator(self, data_dir, batch_size, input_size, hm_size, normalize, category,gaussian_variance, joints_num, center_radius, sample_set ):
        '''generate data and label '''
        while True:
            img_batch= np.zeros((batch_size, input_size, input_size, 3), dtype=np.float32)
            hm_batch=np.zeros((batch_size, hm_size, hm_size, joints_num+1), dtype= np.float32)
            cmap_batch=np.zeros((batch_size, input_size, input_size, 1), dtype= np.float32)
            bx_batch=[]
            for i in range(batch_size):
                if sample_set == 'train':
                    random_index=np.random.randint(0,len(self.train_set_img))
                    img_name= self.train_set_img[random_index]
                    joints = self.train_set_joints[random_index]
                    bounding_box= self.train_set_bx[random_index]
                elif sample_set == 'valid':
                    random_index= np.random.randint(0,len(self.valid_set_img))
                    img_name=self.valid_set_img[random_index]
                    joints = self.valid_set_joints[random_index]
                    bounding_box= self.valid_set_bx[random_index]
                img, h_ratio, w_ratio = self.read_img(data_dir, category ,img_name,input_size, normalize)
                hm = self.generate_hm(joints, input_size,hm_size, h_ratio, w_ratio, gaussian_variance )
                cmap=self.generate_cmap(input_size, center_radius)
                img,hm=self.arguement(img,hm,joints_num,normalize)
                img_batch[i]=img.astype(np.float32)
                hm_batch[i]=hm.astype(np.float32)
                cmap_batch[i]=cmap.astype(np.float32)
                bx_batch.append(bounding_box)
                
            yield img_batch, hm_batch, cmap_batch
            
    def read_file(self, data_dir, category):
        '''将txt中的数据读到list中储存起来，并返回储存train和valid的list '''
        path_train= data_dir + '/' + category + '/' + category + '_train.txt'
        path_valid= data_dir + '/' + category + '/' + category + '_valid.txt'
        train_set_img, train_set_joints, train_set_bx=self.read(path_train)
        valid_set_img,valid_set_joints, valid_set_bx=self.read(path_valid)
        return train_set_img, train_set_joints, train_set_bx, valid_set_img, valid_set_joints, valid_set_bx

    def read(self,txt_path,):
        train_set_img=[]
        train_set_joints=[]
        train_set_bx=[]
        f=open(txt_path, 'r')
        content=f.readlines()
        for row in content:
            row_content=row.strip('\n').split(' ')
            # print(row_content)
            img_name=row_content[0]
            joints=[int(s) for s in row_content[2:]]
            bounding_box=self.compute_box(joints)
            train_set_img.append(img_name)
            train_set_joints.append(joints)
            train_set_bx.append(bounding_box)
        return train_set_img, train_set_joints, train_set_bx
    
    def compute_box(self,joints,ratio=0.3):
        '''直接计算扩展后的bx'''
        bounding_box=[]
        joints_x= joints[::2]#取出所有关节点坐标中的第一个值
        joints_y=joints[1::2]#取出所有关节点坐标中的第二个值
        while -1 in joints_x:
            joints_x.remove(-1)
            joints_y.remove(-1)
        x_min=min(joints_x)
        y_min=min(joints_y)
        x_max=max(joints_x)
        y_max=max(joints_y)
        height=x_max-x_min
        width=y_max-y_min
        x_min=x_min-ceil(height*ratio/2)
        y_min=y_min-ceil(width*ratio/2)
        if x_min < 0:
            x_min=0
        if y_min < 0:
            y_min=0
        x_max=x_min + ceil(height*(1+ratio))
        y_max=y_min + ceil(width*(1+ratio))
        
        bounding_box.append(x_min)
        bounding_box.append(y_min)
        bounding_box.append(x_max)
        bounding_box.append(y_max)
        return bounding_box
        
        

    def read_img(self, data_dir, category, img_name, input_size,normalize):
        #dtaa_dir为存放图片的位置，根据各自存图片的位置进行相应改动
        data_dir='/data/yuwei/research/Tianchi/Project/data/train_data'
        img_path= data_dir + '/'+ img_name
        #print(img_name)
        src_img= cv2.imread(img_path)
        #print(src_img)
        src_h=src_img.shape[0]
        src_w=src_img.shape[1]
        # if bx[2]> src_h:
            # bx[2] = src_h
        # if bx[1] > src_w:
            # bx[3] =src_w
        # cv2.rectangle(src_img, (bx[0],bx[1]),(bx[2],bx[3]),(0,0,255),thickness=2)
        # cv2.imwrite('./result/'+ img_name.split('/')[-1],src_img)
        img=cv2.resize(src_img,(input_size,input_size),interpolation=cv2.INTER_CUBIC)
        if normalize:
            img= img/255.0 - 0.5
        else:
            img = img -128.0
        img=np.asarray(img)
        h_ratio=input_size/src_h
        w_ratio=input_size/src_w

        return img, h_ratio, w_ratio

    def generate_hm(self, joints, input_size,hm_size, h_ratio, w_ratio, gaussian_variance ):
        resize_ratio= input_size/hm_size
        h_ratio=resize_ratio/h_ratio
        w_ratio=resize_ratio/w_ratio
        hm=[]
        hm_bg= np.ones(shape=(hm_size,hm_size))
        num_joints=int(len(joints)/2)
        for i in range(num_joints):
            center_x = joints[i*2] // w_ratio
            center_y=joints[i*2+1] // h_ratio
            sub_hm= self.make_gaussian(hm_size, gaussian_variance, center_x, center_y)
            hm.append(sub_hm)
            #背景对应的热图
            hm_bg=hm_bg-sub_hm
        hm.append(hm_bg)
        hm = np.asarray(hm)
        hm=np.transpose(hm, (1,2,0))

        return hm
        
    def generate_cmap(self,input_size, center_radius):
        center_map = np.zeros((input_size, input_size))
        c_x=input_size/2
        c_y=input_size/2
        for x_p in range(input_size):
            for y_p in range(input_size):
                dist_sq = (x_p - c_x) * (x_p - c_x) + \
                          (y_p - c_y) * (y_p - c_y)
                exponent = dist_sq / 2.0 / center_radius / center_radius
                center_map[y_p, x_p] = np.exp(-exponent)
                
        return np.reshape(center_map,(input_size,input_size,1))
        

    def make_gaussian(self, hm_size, gaussian_variance, center_x, center_y):
        x = np.arange(0, hm_size, 1, float)
        y = x[:, np.newaxis]
        if center_x is None:
            x0 = y0 = size // 2
        else:
            x0 = center_x
            y0 = center_y
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / gaussian_variance/ gaussian_variance)

    def arguement(self, img, hm, joints_num, normalize):
        if normalize:
            img=np.uint8((img+0.5)*255.0)
        else:
            img=np.uint8(img + 128.0)
        if random.choice([0, 1]):
            max_rotation=15
            r_angle = np.random.randint(-1 * max_rotation, max_rotation)
            # img = transform.rotate(img, r_angle, preserve_range=True)
            # hm = transform.rotate(hm, r_angle)
            img_=Image.fromarray(img)
            img_=img_.rotate(r_angle)
            img=np.array(img_)
            hm_=np.zeros((46, 46, joints_num+1),np.float32)
            for channel in range(joints_num+1):
                hm_[:,:,channel]=np.array(Image.fromarray(hm[:,:,channel]).rotate(r_angle))
            hm=hm_
        if random.choice([0,1]):
            img_=Image.fromarray(img)
            hm_=np.zeros((46, 46, joints_num+1),np.float32)
            #不同的类别需要相应改动
            blouse_Flip_channel=[1,0,3,2,4,6,5,8,7,11,12,9,10,13]
            for channel in range(joints_num+1):
                hm_[:,:,channel]=np.array(Image.fromarray(hm[:,:,blouse_Flip_channel[channel]]).transpose(Image.FLIP_LEFT_RIGHT))
            img_=img_.transpose(Image.FLIP_LEFT_RIGHT)
            img=np.array(img_)
            hm=hm_
        
        # if random.choice([0,0,0,0,1]):
            # img_=Image.fromarray(img)
            # brightness=random.choice([0.8,1.5,1.8])
            # img_bri= ImageEnhance.Brightness(img_).enhance(brightness)
            # img=np.array(img_bri)
        # if random.choice([0,0,0,0,1]):
            # img_=Image.fromarray(img)
            # color_factor=random.choice([0.8,1.5,1.8])
            # img_color= ImageEnhance.Color(img_).enhance(color_factor)
            # img=np.array(img_color)
        # if random.choice([0,0,0,0,1]):
            # img_=Image.fromarray(img)
            # contrast_factor=random.choice([0.8,1.5,1.8])
            # img_contrast= ImageEnhance.Contrast(img_).enhance(contrast_factor)
            # img=np.array(img_contrast)
        if normalize:
            img=img/255.0 - 0.5
        else:
            img=img - 128.0
        return img, hm