import torch
from NN import NeuralNet
from dataset import ContourDataset 
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset import format_ax
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from createdata import X,Y

def detect_peaks(pred, target):
    pred *= -1
    target *= -1

    #im = img_as_float(pred)
    #coordinates = peak_local_max(im, min_distance=20)
    
    coordinates = peak_local_max(pred, min_distance=20)
    
    # display results
    fig, axs = plt.subplots(1, 3)

    pc = axs[0].pcolormesh(X, Y, target)
    format_ax(axs[0], pc, fig)
    axs[0].set_title('Target')

    pc = axs[1].pcolormesh(X, Y, pred)
    format_ax(axs[1], pc, fig)
    axs[1].set_title('Prediction')

    pc = axs[2].pcolormesh(X, Y, pred)
    axs[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    format_ax(axs[2], pc, fig)
    axs[2].set_title('Pred. internal vertices')


    
    # ax1.imshow(target, cmap=plt.cm.gray)
    # ax1.axis('off')
    # ax1.set_title('Target')

    # ax2.imshow(pred, cmap=plt.cm.gray)
    # ax2.axis('off')
    # ax2.set_title('Prediction')

    # ax3.imshow(pred, cmap=plt.cm.gray)
    # ax3.autoscale(False)
    # ax3.plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    # ax3.axis('off')
    # ax3.set_title('Pred. internal vertices')

    fig.tight_layout()

    plt.show()
    return coordinates

# load trained model
config = torch.load("model.pth")
model = NeuralNet(device="cpu")
model.load_state_dict(config["state_dict"])
model.eval()

# get dummy data 
test_data = ContourDataset(split="test")
test_loader = DataLoader(test_data, batch_size=1)
it = iter(test_loader)
Data = next(it)
Data = next(it)


df, sdf, Ni = Data

df_pred, df_norm, df_x, df_y = model(sdf,Ni) #.detach().cpu().numpy()
df_pred = df_pred.detach().cpu().numpy().squeeze()
target_df = df.squeeze().detach().cpu().numpy()

peaks = detect_peaks(df_pred, target_df)
print(peaks)
