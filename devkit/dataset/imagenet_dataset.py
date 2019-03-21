from torch.utils.data import Dataset
from PIL import Image
import torch

def pil_loader(filename):
    with Image.open(filename) as img:
        img = img.convert('RGB')
    return img
 
class ImagenetDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        print("building dataset from %s"%meta_file)
        self.num = len(lines)
        self.metas = []
        for line in lines:
            path, cls = line.rstrip().split()
            self.metas.append((path, int(cls)))
        print("read meta done")
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, idx):
        filename = self.root_dir + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]

        img = pil_loader(filename)
        
        ## transform
        if self.transform is not None:
            img = self.transform(img)
        return img, cls


class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [0.4009, 0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor