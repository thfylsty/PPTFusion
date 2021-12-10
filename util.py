import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DatasetFromFolder(Dataset):
    def __init__(self, paths):
        super(DatasetFromFolder, self).__init__()

        self.target_img_path = glob.glob(os.path.join(paths, '*.*'))

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        paths = self.target_img_path[index]
        img = Image.open(paths).convert('L').resize((256,256))
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.target_img_path)

class DatasetFromFolder_viir(Dataset):
    def __init__(self, paths):
        super(DatasetFromFolder_viir, self).__init__()

        self.target_img_path = glob.glob(os.path.join(paths, '*.*'))
        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        paths = self.target_img_path[index]
        img_vi = Image.open(paths).resize((256,256))
        img_ir = Image.open(paths.replace('vi','ir')).resize((256,256))
        img_vi = self.transform(img_vi)
        img_ir = self.transform(img_ir)

        return img_vi,img_ir

    def __len__(self):
        return len(self.target_img_path)