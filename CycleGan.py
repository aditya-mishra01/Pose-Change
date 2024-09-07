
from torch import nn
import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader,Dataset
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
torch.manual_seed(0)

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):


    image_tensor = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

# class ImageDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.transform = transform
#         self.files = sorted(glob.glob(os.path.join(root, '*.*')))
#         self.size=self.size = len(self.files)
#         assert len(self.files) > 0

#     def __getitem__(self, index):
#         item = self.transform(Image.open(self.files[index]))

#         if item.shape[0] != 3:
#             item = item.repeat(3, 1, 1)

#         return (item - 0.5) * 2

#     def __len__(self):
#         return len(self.files)
def add_noise_to_center(image, noise):
    image_height, image_width = image.shape[1], image.shape[2]
    noise_height, noise_width = noise.shape[1], noise.shape[2]

    start_y = (image_height - noise_height) // 2
    start_x = (image_width - noise_width) // 2

    image[:, start_y:start_y+noise_height, start_x:start_x+noise_width] += noise
    return image
def process_batch(x_batch, gt_batch):
    noisy_batch = []

    for i in range(x_batch.size(0)):
        x = x_batch[i]
        gt = gt_batch[i]
        gt_np = gt.detach().cpu().numpy()
        gt_np_padded = np.pad(gt_np, (0, 7), 'constant')
        array_7x7 = np.reshape(gt_np_padded, (7, 7))
        noise_3x7x7 = np.stack([array_7x7] * 3, axis=0)
        x_np = x.detach().cpu().numpy()
        noisy_x_np = add_noise_to_center(x_np.copy(), noise_3x7x7)
        noisy_x = torch.tensor(noisy_x_np).to("cuda")
        noisy_batch.append(noisy_x)

    
    noisy_batch = torch.stack(noisy_batch)
    return noisy_batch

class ResidualBlock(nn.Module):


    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.instancenorm = nn.InstanceNorm2d(input_channels)
        self.activation = nn.ReLU()

    def forward(self, x):

        original_x = x.clone()
        x = self.conv1(x)
        x = self.instancenorm(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instancenorm(x)
        return original_x + x

class ContractingBlock(nn.Module):

    def __init__(self, input_channels, use_bn=True, kernel_size=3, activation='relu'):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=kernel_size, padding=1, stride=2, padding_mode='reflect')
        self.activation = nn.ReLU() if activation == 'relu' else nn.LeakyReLU(0.2)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels * 2)
        self.use_bn = use_bn

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x

class ExpandingBlock(nn.Module):

    def __init__(self, input_channels, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        if use_bn:
            self.instancenorm = nn.InstanceNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        if self.use_bn:
            x = self.instancenorm(x)
        x = self.activation(x)
        return x
class FeatureMapBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=7, padding=3, padding_mode='reflect')

    def forward(self, x):

        x = self.conv(x)
        return x
class ChangedFeature(nn.Module):
    def __init__(self, feature_map_channels, matrix_size, hidden_dim):
        super(ChangedFeature, self).__init__()

        self.query_layer = nn.Linear(matrix_size, hidden_dim)
        self.key_layer = nn.Linear(feature_map_channels * 32 * 32, hidden_dim)
        self.value_layer = nn.Linear(feature_map_channels * 32 * 32, hidden_dim)
        self.flat=nn.Flatten()
        self.unflat=nn.Sequential(
            nn.Linear(hidden_dim,feature_map_channels * 32 * 32),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (feature_map_channels, 32, 32))
        )



    def forward(self, feature_map, matrix):
        feature_map_flat=self.flat(feature_map)
        query = self.query_layer(matrix)
        key = self.key_layer(feature_map_flat)
        value = self.value_layer(feature_map_flat)


        attention_scores = torch.matmul(query, key.transpose(0, 1))


        attention_weights = torch.softmax(attention_scores, dim=-1)


        attended_feature_map = torch.matmul(attention_weights, value)

        attended_feature_map = self.unflat(attended_feature_map)

        return attended_feature_map

class Generator(nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64):
        super(Generator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels,activation='LR')
        self.contract2 = ContractingBlock(hidden_channels * 2,activation='LR')
        self.contract3= ContractingBlock(hidden_channels * 4,activation='LR')
        res_mult = 8
        self.res0 = ResidualBlock(hidden_channels * res_mult)
        self.res1 = ResidualBlock(hidden_channels * res_mult)
        self.res2 = ResidualBlock(hidden_channels * res_mult)
        self.res3 = ResidualBlock(hidden_channels * res_mult)
        self.res4 = ResidualBlock(hidden_channels * res_mult)
        self.res5 = ResidualBlock(hidden_channels * res_mult)
        self.res6 = ResidualBlock(hidden_channels * res_mult)
        self.res7 = ResidualBlock(hidden_channels * res_mult)
        self.res8 = ResidualBlock(hidden_channels * res_mult)
        self.expand1=ExpandingBlock(hidden_channels * 8)
        self.expand2 = ExpandingBlock(hidden_channels * 4)
        self.expand3 = ExpandingBlock(hidden_channels * 2)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        # self.change=ChangedFeature(hidden_channels*8,42,128)

    def forward(self, x,gt):
        x0=process_batch(x,gt)

        x0 = self.upfeature(x0)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3=self.contract3(x2)
        # x3=self.change(x3,gt)
        x4 = self.res0(x3)
        x5 = self.res1(x4)
        x6 = self.res2(x5)
        x7 = self.res3(x6)
        x8 = self.res4(x7)
        x9 = self.res5(x8)
        x10 = self.res6(x9)
        x11 = self.res7(x10)
        x12 = self.res8(x11)
        # x10=self.change(x10,gt)
        x13=self.expand1(x12)
        x14 = self.expand2(x13)
        x15 = self.expand3(x14)
        xn = self.downfeature(x15)
        return self.tanh(xn)

class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super(Discriminator, self).__init__()
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract1 = ContractingBlock(hidden_channels, use_bn=False, kernel_size=4, activation='lrelu')
        self.contract2 = ContractingBlock(hidden_channels * 2, kernel_size=4, activation='lrelu')
        self.contract3 = ContractingBlock(hidden_channels * 4, kernel_size=4, activation='lrelu')
        self.final = nn.Conv2d(hidden_channels * 8, 1, kernel_size=1)

    def forward(self, x):
        x0 = self.upfeature(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        xn = self.final(x3)
        return xn