import numpy as np
import cv2
import math

def visiaul(img, hm, input_size, normalize, joints_num):
    if normalize:
        img = (img + 0.5) 
    else:
        img= img + 128.0
    joints_hm = hm[: , : , 0:joints_num]
    joints_hm=cv2.resize(joints_hm, (input_size, input_size))
    joints_hm=np.amax(joints_hm, axis=2)
    joints_hm= np.reshape(joints_hm, (input_size,input_size,1))
    joints_hm= np.repeat(joints_hm, 3, axis=2)
    
    if normalize:
        blend_img = 0.5 * img + 0.5* joints_hm
    else:
        blend_img = 0.5*img/255.0 + 0.5* joints_hm
    blend_img= (blend_img*255).astype(np.uint8)
    return blend_img   
    
def visualize_result(test_img, stage_heatmap_np, joints, hmap_size,joint_color_code):
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:joints].reshape(
        (hmap_size,hmap_size, joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    joint_coord_set = np.zeros((joints, 2))

    # Plot joint colors
    for joint_num in range(joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord_set[joint_num, :] = [joint_coord[0], joint_coord[1]]

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    
    return test_img
    
def gaussian_img(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map

    
def read_image(file_path,boxsize, type):
    # from file
    file= file_path
    oriImg = cv2.imread(file)
    # from webcam
    if oriImg is None:
        print('oriImg is None')
        return None

    scale = boxsize / (oriImg.shape[0] * 1.0)
    imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    output_img = np.ones((boxsize, boxsize, 3)) * 128

    img_h = imageToTest.shape[0]
    img_w = imageToTest.shape[1]
    if img_w < boxsize:
        offset = img_w % 2
        # make the origin image be the center
        output_img[:, int(boxsize / 2 - math.floor(img_w / 2)):int(
            boxsize / 2 + math.floor(img_w / 2) + offset), :] = imageToTest
    else:
        # crop the center of the origin image
        output_img = imageToTest[:,
                     int(img_w / 2 - boxsize / 2):int(img_w / 2 + boxsize / 2), :]
    return output_img
    
def chose_img_test(path,test_num):
    test_img=[]
    f=open(path,'r')
    img_list=f.readlines()
    num_img=len(img_list)
    for i in range(test_num):
        random_choice=np.random.randint(0, num_img)
        img = img_list[random_choice].split(' ')[0]
        test_img.append(img)
    return test_img
        
    
    
    