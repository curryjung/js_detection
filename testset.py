import sys
import glob
import shutil
import os


def make_testset(name = 'test', path=None):
    
    sample_rate = 0.1
    
    neg_path = os.path.join(path,'neg')
    pos_path = os.path.join(path,'pos')

    neg_test_path = os.path.join(path,'neg_'+name)
    pos_test_path = os.path.join(path,'pos_'+name)

    if not os.path.isdir(neg_test_path):
        print('no neg dir')
        os.mkdir(neg_test_path)
    
    if not os.path.isdir(pos_test_path):
        print('no pos dir')
        os.mkdir(pos_test_path)
    
    neg_list = glob.glob(neg_path + '/*.png')
    pos_list = glob.glob(pos_path + '/*.png')

    neg_sample_rate = round( 1/sample_rate )
    pos_sample_rate = round( 1/sample_rate )

    for idx in range(0,len(neg_list),neg_sample_rate):
        neg_sample = neg_list[idx]
        file_name = os.path.split(neg_sample)[1] # image 이름만 추출
        shutil.copy(neg_sample,os.path.join(neg_test_path,file_name)) #test dir 로 copy
        os.unlink(neg_sample) #copy 한 image 삭제

    for idx in range(0,len(pos_list),pos_sample_rate):
        pos_sample = pos_list[idx]
        file_name = os.path.split(pos_sample)[1] # image 이름만 추출
        shutil.copy(pos_sample,os.path.join(pos_test_path,file_name)) #test dir 로 copy
        os.unlink(pos_sample) #copy 한 image 삭제

    

if __name__ == '__main__':

    name = 'val'
    path = '/media/jaeseok/3F193D906CC86E6A/researches/js_detection/dataset'
    make_testset(name = name, path = path)