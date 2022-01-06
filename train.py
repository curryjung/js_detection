import torch
from model import FaceDetector
from dataset import JSdataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import os
import tqdm

import time
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 10
epoch = 0
learning_rate = 1e-3
print_every = 10
evaluate_every = 10
out_dir = "./out"
total_epoch=5


logger = SummaryWriter(os.path.join(out_dir, 'log'))

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def train(model, optimizer, criterion, train_loader, test_loader, total_epoch):

    model.train()

    it = 0

    t0b = time.time()
    for epoch in range(total_epoch):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.type(torch.FloatTensor)
            labels = labels.to(device)

            loss = 0

            #initialize gradient
            optimizer.zero_grad()
            #forward propagation
            output = model(images)
            #calculate loss
            loss = criterion(output.squeeze(1),labels)
            #backward proopagation
            loss.backward()
            #update parameters
            optimizer.step()


            loss_dicts = {"Loss": loss }
            logger.add_scalars("train_loss",loss_dicts,it)

            if print_every > 0 and (it % print_every) == 0:
                print('[Epoch %02d] loss: %.4f it %03d, time%.3f' % (epoch,loss, it, time.time()-t0b))
            
            it+=1

def get_pred(pytorch_model,x):
    x = torch.Tensor(x).to(device)
    output = pytorch_model(x)
    output = torch.round(output, axis=1)
    output = output.cpu().numpy()
    return output    

def get_accuracy(pred,label):
    return torch.sum(pred == label).item()/len(label)


def evaluate():
    pass
    return NotImplementedError('please implement later')

def run():

    train_dataset = JSdataset('/data/js_detection/dataset',train=True,transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5244, 0.5401, 0.5397),(0.2569, 0.2638, 0.2730))
    ]))

    test_dataset = JSdataset('/data/js_detection/dataset',train=False,transform =transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize((0.5244, 0.5401, 0.5397),(0.2569, 0.2638, 0.2730))
    ]))



    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    model = FaceDetector()
    model.to(device)

    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()
    train(model=model, optimizer=optimizer, criterion=criterion, train_loader=train_loader,test_loader=test_loader,total_epoch=total_epoch)



if __name__=='__main__':
    run()

else:
    pass