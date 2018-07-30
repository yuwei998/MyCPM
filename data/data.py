import csv
import os
import numpy as np

def split_txt(category, file_name):
    '''以0.8，0.1,0.1 将数据分为训练、验证、测试'''
    file_path= category + '/' + file_name
    train_path=category + '/' + category + '_train.txt'
    valid_path=category + '/' + category + '_valid.txt'
    test_path=category + '/' + category + '_test.txt'
    
    test_num=0
    train_num=0
    valid_num=0
    
    f=open(file_path, 'r')
    f_train=open(train_path, 'w')
    f_valid=open(valid_path, 'w')
    f_test=open(test_path, 'w')
    content= f.readlines()
    for row in content:
        random_num=np.random.randint(0,100)
        write_content=row
        if random_num <= 10:
            f_test.write(write_content)
            test_num=test_num+1
        elif random_num <= 20:
            f_valid.write(write_content)
            valid_num=valid_num+1
        else:
            f_train.write(write_content)
            train_num=train_num+1
    
    print('num of train data: ' + str(train_num))
    print('num of test data: ' + str(test_num))
    print('num of valid data: ' + str(valid_num))


def csv2txt(category, joints, csv_path):
    file_name=category + '.txt'
    file_path=os.path.join(category)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path= file_path + '/' + file_name
    data_num=0
    f=open(file_path, 'w')
    for item in csv_path:
        #csv中定义的关节点的顺序
        csv_joints=[]
        csv_contenten=csv.reader(open(item))
        for row in csv_contenten:
            if 'image_id' in row:
                csv_title=row
            if category in row:
                data_num=data_num+1
                write_content = row[0] + ' ' + row[1]
                for joint in joints:
                    joint_index= csv_title.index(joint)
                    #将'1_1_1'格式的数据转为'1 1'
                    joint_value=str_transfer(row[joint_index])
                    write_content= write_content + ' ' + joint_value
                write_content= write_content + '\n'
                f.write(write_content)
    print('*************' + category + '**************')
    print('num all data ' + str(data_num))
    return file_name
                
def str_transfer(str):
    ''' 类似于1_1_1的数据转为'1 1'  '''
    join_content=str.split('_')[0: -1]
    str=' '.join(join_content)
    return str
    

       

def main():

    csv_path1='/data/yuwei/research/Tianchi/Project/data/train_data/train.csv'
    csv_path2='/data/yuwei/research/Tianchi/Project/data/train_data/train_a.csv'
    csv_path3='/data/yuwei/research/Tianchi/Project/data/train_data/train_b.csv'
    csv_path=[csv_path1, csv_path2, csv_path3]

    category_list=['blouse', 'skirt', 'dress', 'outwear', 'trousers']

    joints_dic={'blouse_list': ['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'center_front', 'armpit_left', 
                                'armpit_right', 'top_hem_left', 'top_hem_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out'] , 
               'dress_list':  ['neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 
                             'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 
                             'cuff_right_out', 'hemline_left', 'hemline_right'], 
               'skirt_list':  ['waistband_left', 'waistband_right', 'hemline_left', 'hemline_right'],  
               'outwear_list':['neckline_left', 'neckline_right', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 
                               'waistline_left', 'waistline_right', 'top_hem_left', 'top_hem_right', 'cuff_left_in', 'cuff_left_out', 
                               'cuff_right_in', 'cuff_right_out'], 
               'trousers_list':['waistband_left', 'waistband_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out'] }
           
    for category in category_list:
        joints_list=joints_dic[category+'_list']
        
        #生成所有数据对应的txt文件并保存，返回保存的txt的名字
        file_name=csv2txt(category, joints_list,csv_path)
        
        #将所有的数据分为train, valid, test三部分
        split_txt(category, file_name)

if __name__ == '__main__':
    main()
