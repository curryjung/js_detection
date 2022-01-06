import torch
import torch.nn as nn
import torch.nn.functional as F



class FaceDetector(nn.Module):
    def __init__(self):
        
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(8,16,kernel_size=3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=7),
            nn.ReLU()
        )
        
        self.fc1 = nn.Linear(128,128)
        self.fc2 = nn.Linear(128,1)
        
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
        
        

    def forward(self,x):
        x=self.model(x)
        x = x.view(x.shape[0],-1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
    def __repr__(self) -> str:
        return "FaceDetector()"
    