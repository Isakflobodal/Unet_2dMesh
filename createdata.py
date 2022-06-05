import numpy as np
import matplotlib.pyplot as plt
import random
import pygmsh
import os
import torch
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyvista as pv
from PIL import Image, ImageDraw
import skfmm

def WeightedRandom():
    list_random = [random.uniform(0.5,0.6), random.uniform(0.6,0.7), random.uniform(0.7,0.8),random.uniform(0.8,0.9),random.uniform(0.9,1.0)]
    weights = (5,5,25,30,35)
    return random.choices(list_random, weights, k=1)[0]

def ScaleAndSenter(contour):
    # Find max dist between pts
    max_dist = 0
    for a, b in combinations(np.array(contour),2):
        cur_dist = np.linalg.norm(a-b)
        if cur_dist > max_dist:
            max_dist = cur_dist
   
    # center and normalize contour
    center = np.mean(contour,axis=0)      
    contour = contour - center                 
    contour /= max_dist                     
    return contour

def GetRandomContour(N):
    l = 1.0
    b = WeightedRandom()
    contour = np.array([[-l/2,-b/2,0],[l/2,-b/2,0],[l/2,b/2,0],[-l/2,b/2,0]])

    contour = ScaleAndSenter(contour=contour)
    l = abs(contour[0][0])*2
    b = abs(contour[0][1])*2

    nl = max(round(np.sqrt(N*l/b))+1,2)
    nb = max(round(N/nl)+1,2)


    ls = np.linspace(-l/2,l/2,nl)
    bs = np.linspace(-b/2,b/2,nb)

    X,Y = np.meshgrid(ls,bs)
    pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(nl*nb,0))])
    
    
    mesh = pv.StructuredGrid()
    mesh.points = pts
    mesh.dimensions = [nl,nb,1]
    #mesh.plot(show_edges=True, show_grid=True, cpos = 'xy')

    return pts, contour, mesh

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
    return sdf

# Returns a distance field
def GetDF(pts):
    distances = np.sqrt((pts[:,0][:,None] - X.ravel())**2 + (pts[:,1][:,None] - Y.ravel())**2)
    min_dist = distances.min(axis=0)
    return min_dist.reshape([dim,dim])

# Create contours, mesh and distance field
def CreateData(dataType,i, N):
    path = f'./data/{dataType}/{i}'
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)     

    mesh_pts, P, mesh = GetRandomContour(N=N)        
    pv.save_meshio(f'{path}/mesh.vtk', mesh=mesh)


    sdf = GetSignedDistance(P[:,:-1])
    df = GetDF(mesh_pts[:,:-1])

    sdf = torch.from_numpy(np.array(sdf)).view(1,dim,dim).float()
    df = torch.from_numpy(np.array(df)).view(1,dim,dim).float()
    
    data = {
        "P": P,
        "mesh_pts": mesh_pts,
        "sdf": sdf,
        "df": df
        }

    torch.save(data,f'{path}/data.pth')
    return df, sdf

# create training data for testing, training and validation
def CreateDataMain(N):
    for i in range(training_samples):
        if i < testing_samples:
            CreateData("test",i,N)
        if i < training_samples:
            df, sdf = CreateData("train",i, N)
            df_list.append(df)
            sdf_list.append(sdf)
        if i < validation_samples:
            CreateData("validation",i, N)
    print('Data sets created!')

# Returns the inner points in the mesh
def GetInnerPts(mesh_pts, contour_pts):
    inner_pts = []
    for pt in mesh_pts:
        contour_pts = torch.DoubleTensor(contour_pts)
        pt = torch.DoubleTensor(pt)
        res = torch.isclose(pt[:2], contour_pts)
        res = res[:,0] & res[:,1]
        if not (res.sum()>0):
            inner_pts.append(pt)
    return inner_pts

def format_ax(ax, pc,fig, loc='bottom'):
    ax.axis('scaled')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size='5%', pad=0.2)
    fig.colorbar(pc, cax=cax, orientation='horizontal')

# Plots the distance field
def PlotDistanceField(df,sdf):
    fig, axs = plt.subplots(1,2, figsize=(12,8))

    pc = axs[0].pcolormesh(X,Y, df, cmap='viridis')
    format_ax(axs[0], pc, fig)
    #axs[0].set_title('Distance field')
    

    pc = axs[1].pcolormesh(X,Y, sdf, cmap='hsv')
    format_ax(axs[1], pc, fig)
    #axs[1].set_title('Distance field')


    plt.show()


# Grid size
dim = 256
min_xy, max_xy = -0.55, 0.55
step = (max_xy - min_xy)/dim
xs = np.linspace(min_xy,max_xy,dim)
ys = np.linspace(min_xy,max_xy,dim)
X,Y = np.meshgrid(xs,ys)
pts = np.array([list(pair) for pair in zip(X.flatten(),Y.flatten(), np.full(dim**2,0))])

# Hyperparameters for data
testing_samples = 1000
training_samples = 10000
validation_samples= 1000
N = 20

df_list = []
sdf_list = []


if __name__ == "__main__":
    CreateDataMain(N=N)
    PlotDistanceField(df_list[0].cpu().squeeze().detach().numpy(),sdf_list[0].cpu().squeeze().detach().numpy())