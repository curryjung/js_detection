from dataset import JSdataset
from torch.utils.data import DataLoader


def cal_mean_val():
    W = 640
    H = 480

    js_dataset = JSdataset('/data/js_detection/dataset',transform=transforms.ToTensor())
    loader = DataLoader(js_dataset,batch_size=10,num_worker=0,shuffle=False)
    
    mean = 0.
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples,images.size(3),-1)
        mean += images.mean(2).sum(0)
    
    mean = mean/len(loader.dataset)

    var = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(3), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*W*H))

if __name__='__main__':
    cal_mean_val()