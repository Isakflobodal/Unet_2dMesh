import matplotlib.pyplot as plt
from NN import NeuralNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.autograd as ag
from dataset import format_ax
from createdata import X,Y
import tqdm
import wandb
from torchvision.utils import make_grid


@torch.no_grad()
def no_grad_loop(data_loader, model, df_loss_value=0, epoch=2, device="cuda", batch_size = 64):
    valid_loss = 0
    valid_loss_df = 0
    valid_loss_grad = 0
    cnt = 0
    for i, (df, sdf) in enumerate(data_loader):

        # transfer data to device
        df = df.to(device)
        sdf = sdf.to(device)

        with autocast():
            df_pred, df_norm, df_x, df_y = model(sdf)
            loss_df = F.l1_loss(df_pred, df)
            loss_grad = (df_norm-1)**2
            loss_SDF = loss_grad.mean()
        loss = loss_df + loss_SDF*1e-07
    
        valid_loss_df += loss_df
        valid_loss += loss
        valid_loss_grad += loss_SDF
        cnt += 1

        if i == len(data_loader) -1:
            
            pad = 2
            case = 0

            X_, Y_ = X[pad:-pad,pad:-pad], Y[pad:-pad,pad:-pad]

            fig, axs = plt.subplots(2,4, figsize=(32,16))

            pc = axs[0][0].pcolormesh(X, Y, sdf[case].cpu().squeeze().detach().numpy(), cmap='terrain')
            format_ax(axs[0][0], pc, fig)
            axs[0][0].set_title('sdf terrain map')

            pc = axs[0][1].pcolormesh(X, Y, sdf[case].cpu().squeeze().detach().numpy())
            format_ax(axs[0][1], pc, fig)
            axs[0][1].set_title('sdf color map')

            pc = axs[0][2].pcolormesh(X, Y, df[case].cpu().squeeze().detach().numpy())
            format_ax(axs[0][2], pc, fig)
            axs[0][2].set_title('df gt')

            pc_x = axs[0][3].pcolormesh(X, Y, df_pred[case].cpu().squeeze().detach().numpy())
            format_ax(axs[0][3], pc_x, fig)
            axs[0][3].set_title('df pred')

            pc_y = axs[1][0].pcolormesh(X_, Y_, df_x[case].cpu().squeeze().detach().numpy())
            format_ax(axs[1][0], pc_y, fig)
            axs[1][0].set_title(r'$\partial_x df$')

            pc_y = axs[1][1].pcolormesh(X_, Y_, df_y[case].cpu().squeeze().detach().numpy())
            format_ax(axs[1][1], pc_y, fig)
            axs[1][1].set_title(r'$\partial_y df$')

            pc_norm = axs[1][2].pcolormesh(X_, Y_, df_norm[case].cpu().squeeze().detach().numpy())
            format_ax(axs[1][2], pc_norm, fig)
            axs[1][2].set_title(r'$||\nabla df||$')

            pc_loss = axs[1][3].pcolormesh(X_, Y_, loss_grad[case].cpu().squeeze().detach().numpy())
            format_ax(axs[1][3], pc_loss, fig)
            axs[1][3].set_title(r'$(||\nabla df||-1)^2$')
            
            fig.tight_layout()
            
            plt.savefig('train.png')
            wandb.log({"images": wandb.Image("train.png")}, commit=False)
            plt.close(fig)
            plt.close('all')

    return valid_loss_df/cnt, valid_loss/cnt, valid_loss_grad/cnt

def train(model: NeuralNet, num_epochs, batch_size, train_loader, test_loader, validation_loader, df_loss_value=0, learning_rate=1e-3, device="cuda"):

    valid_loss_df, valid_loss, valid_loss_SDF = no_grad_loop(validation_loader, model, epoch=0, device="cuda", batch_size=batch_size)
    curr_lr =  learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=4)

    # early stopping
    last_loss = 100000000
    best_loss = 100000000
    patience = 6
    trigger_times = 0

    # training loop
    iter = 0
    training_losses = {
        "train": {},
        "valid": {}
    }
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        loader = tqdm.tqdm(train_loader)
        for df, sdf in loader:  

            # transfer data to device
            df = df.to(device)
            sdf = sdf.to(device)
            
            # Forward pass
            with autocast():
                df_pred, df_norm, df_x, df_y = model(sdf) 
                loss_df = F.l1_loss(df_pred, df)
                loss_grad: torch.Tensor = (df_norm-1)**2
                loss_SDF = loss_grad.mean()
                loss = loss_df + loss_SDF*1e-07
            loader.set_postfix(df = loss_df.item(), sdf = loss_SDF.item())
            
            # Backward and optimize
            optimizer.zero_grad()               # clear gradients
            scaler.scale(loss).backward()       # calculate gradients

            # grad less than 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)

            scaler.step(optimizer)
            scaler.update()    
            iter += 1  

            training_losses["train"][iter] = loss.item()
            if (iter+1) %  100 == 0:

                # validation loop
                model = model.eval()
                valid_loss_df, valid_loss, valid_loss_SDF = no_grad_loop(validation_loader, model, df_loss_value, epoch, device="cuda", batch_size=batch_size)

                # early stopping
                if valid_loss > last_loss:
                    trigger_times +=1
                    if trigger_times >= patience:
                        return model
                else:
                    trigger_times = 0
                last_loss = valid_loss

                # save check point
                if valid_loss < best_loss:
                    config = {"state_dict": model.state_dict()}
                    torch.save(config, "model_check_point.pth")
                    best_loss = valid_loss


                scheduler.step(valid_loss)
                curr_lr =  optimizer.param_groups[0]["lr"]
                wandb.log({"valid loss": valid_loss.item(), "valid loss SDF": valid_loss_SDF.item(), "valid loss df": valid_loss_df.item(), "lr": curr_lr}, commit=False)
                model = model.train()
            wandb.log({"train loss": loss.item(), "train loss SDF": loss_SDF.item(), "train loss df": loss_df.item()})

    # test loop
    test_loss_df, test_loss, test_loss_SDF = no_grad_loop(test_loader, model, df_loss_value, device="cuda", batch_size=batch_size)
    print(f'Testloss: df={test_loss_df:.5f}, g={test_loss_SDF:.5f}, tot={test_loss:.5f}')
    
    return model