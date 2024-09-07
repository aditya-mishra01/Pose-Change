import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import glob
import random
from PIL import Image
import os
import o3d
class ImageAndGT(Dataset):
    def __init__(self, root, gt, transform=None):
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root, '*.*')))
        self.gt_files = sorted(glob.glob(os.path.join(gt, '*.*')))
        self.size = min(len(self.files),len(self.gt_files))
        print(len(self.files),len(self.gt_files))
        assert self.size > 0

    def process_gt(self,gt_file):
      pcd=o3d.io.read_point_cloud(gt_file)
      points=np.array(pcd.points)
      points=points.reshape(42)
      points=torch.from_numpy(points)
      points = points.to(torch.float32)
      return(points)

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files[index]))

        if item.shape[0] != 3:
            item = item.repeat(3, 1, 1)

        return (item - 0.5) * 2, self.process_gt(self.gt_files[index])

    def __len__(self):
        return self.size
    

