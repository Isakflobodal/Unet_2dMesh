from cmath import exp
import math
import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
import random
from PIL import Image
import skfmm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from createdata import dim, max_xy, step, X, Y
from itertools import combinations

def GetSignedDistance(contour: np.array):
    # create image where inside is white and outside black
    contour = (contour+max_xy)/(max_xy*2)
    a = (contour*dim).reshape(-1).tolist()
    img = Image.new("RGB", size=[dim, dim], color="black")
    img1 = ImageDraw.Draw(img)
    img1.polygon(a, outline ="white", fill="white")

    # convert image to tensor
    ima = np.array(img)

    # differentiate the inside / outside rigion
    phi = np.int64(np.any(ima[:,:,:3], axis = 2))
    phi = np.where(phi, 0, -1) + 0.5
 
    # compute the signed distance
    sdf = skfmm.distance(phi, dx =step) 
    return torch.from_numpy(np.array(sdf))


def GetDF(pts):
    distances = torch.sqrt((pts[:,0][:,None] - X.ravel())**2 + (pts[:,1][:,None] - Y.ravel())**2)
    min_dist, _ = distances.min(axis=0)
    return min_dist.reshape([dim,dim])

class ContourDataset(Dataset):
    def __init__(self, root_dir="data", split="train"):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split

    def __len__(self):
        return len(self.data)
        #return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"
        data = torch.load(f'{full_path}/data.pth')

        #Pc = torch.from_numpy(data['Pc']).float()
        Pi = torch.from_numpy(data['mesh_pts']).float()
        df = data['df']    
        sdf = data['sdf'] 

        # # rotation
        # if self.split == "train":
        #     phi = random.uniform(0,2*math.pi)
        #     rot_mat = torch.tensor([[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]])

        #     #get signed distance field
        #     Pc_rot = Pc_orig @ rot_mat.T
        #     sdf_rot = GetSignedDistance(Pc_rot)    

        #     #get distance field
        #     Pi_rot = Pi_orig @ rot_mat.T
        #     df_rot = GetDF(Pi_rot)

        #     sdf = sdf_rot.view([1,dim,dim]).float()       # [1, 256,256]
        #     df = df_rot.view([1,dim,dim]).float()         # [1, 256,256]
        
      

        return df, sdf

def format_ax(ax, pc,fig, loc='bottom'):
    ax.axis('scaled')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size='5%', pad=0.4)
    fig.colorbar(pc, cax=cax, orientation='horizontal')


def main():
    dataset = ContourDataset(split="train")
    df, sdf, N_vert = dataset[0]


    # ### PLOT ###
    
    X_,Y_ = X,Y
    fig, axs = plt.subplots(1,2, figsize=(10,10))
    
    # pc = axs[0].pcolormesh(X,Y_,sdf_rot.squeeze().detach().numpy(), cmap='terrain')
    # format_ax(axs[0],pc, fig)
    # axs[0].set_title('sdf_rot')

    pc = axs[0].pcolormesh(X,Y,sdf.squeeze().detach().numpy(), cmap='terrain')
    format_ax(axs[0],pc,fig)
    axs[0].set_title('sdf')

    # pc = axs[2].pcolormesh(X,Y_,df_rot.squeeze().detach().numpy())
    # format_ax(axs[2],pc, fig)
    # axs[2].set_title('df_rot')

    pc = axs[1].pcolormesh(X,Y,df.squeeze().detach().numpy())
    format_ax(axs[1],pc,fig)
    axs[1].set_title('df')
    plt.show()

if __name__ == "__main__":
    main()