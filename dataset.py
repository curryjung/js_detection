import torch
from torch.utils.data import Dataset
import numpy as np
import os, sys, glob
import torchvision
from torch.utils.data import DataLoader
import cv2


class JSdataset(Dataset):
    def __init__(self,path, train=True, transform = None):
        # 객체 선언할때 초기화
        
        # TODO: 구현
        self.path = path


        if train:
            self.neg_path = os.path.join(path,'neg')
            self.pos_path = os.path.join(path,'pos')
        else :
            self.neg_path = os.path.join(path,'neg_test')
            self.pos_path = os.path.join(path,'pos_test')

        self.neg_list = glob.glob(self.neg_path + '/*.png')
        self.pos_list = glob.glob(self.pos_path + '/*.png')
            

        
        self.img_list = self.neg_list + self.pos_list
        self.labels = [0]*len(self.neg_list) + [1]*len(self.pos_list)

        # 이걸 하려면 기본적으로 전체 데이터셋에 대해 평균 분산을 구해서 정규화해야함
        self.transform = transform  # torchvision.transforms 에서 다 있음~
        
    
    
    def __getitem__(self, idx):


        # 인덱스 별 아이템 얻을때 호출
        # iterator, generator --> 이미지같은건 램에 한번에 올리기 어려움
        
        imgpath = self.img_list[idx]
        
        img = cv2.imread(imgpath)
        img = self.transform(img)
        
        label = self.labels[idx]
        
        return img, label

    
    def __len__(self):
        # 전체 데이터셋 길이 --> 데이터로더에서 사용
        return len(self.img_list)
    
    
    
def test():
    dataset = JSdataset('/data/js_detection/dataset')
    print(os.path.abspath(__file__))
    print(len(dataset.img_list))
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch = next(iter(dataloader))
    
    print(batch[0].size(3))
    #print(batch)
    
    assert len(batch[0][0]) == 2
    
    


if __name__ == '__main__':
    test()
    