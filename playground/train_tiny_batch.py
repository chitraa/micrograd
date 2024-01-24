from tiny_network import TinyNN
from data_utils import get_device, TinyDataset, seed_experiment, get_tb_logging_dir
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

    
if __name__ == "__main__":

    seed_experiment()

        # write tb logs, view via 
    log_dir = get_tb_logging_dir("Vanilla-Pytorch-batched-input")
    writer = SummaryWriter(log_dir=log_dir)

    device = get_device()
    print(f"Using {device} device")


    tiny_dataset = TinyDataset()

    # split train and validation dataset
    train_dataset, val_dataset = torch.utils.data.random_split(tiny_dataset, [0.6, 0.4])
    
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    network = TinyNN().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    
    for epoch in range(0, 1000):
        print(f"Epoch {epoch}") 
        
        # TRAINING PHASE
        # zero grad before epoch
        optimizer.zero_grad()

        # loss to accumulate over batches in this epoch
        train_loss = 0.0
            
        # Iterate over the DataLoader
        for i, data in enumerate(train_loader, 0):
      
        # Get inputs
            xs, ys = data
            xs = xs.to(device)
            ys = ys.to(device)

            # 1 a. forward pass
            y_preds = network(xs)
            y_preds = y_preds.squeeze(dim=1)

        
            # 1 b.  calculate loss
            loss = loss_fn(y_preds, ys)
            train_loss += loss.item()
            # print(f"Loss for minibatch in epoch {epoch} is {loss.item()}")

            # 2. backward pass (Calculate grads)
            loss.backward()

            # 3. Update params (optimizer step)
            optimizer.step()
            # print(f"y_preds={y_preds}")

        # average loss for this epoch
        average_train_loss = train_loss / len(train_loader)
        print(f"Train loss for epoch {epoch} is {average_train_loss}")
        writer.add_scalar('Loss/train', average_train_loss, epoch)
    
        # vALIDATION PHASE
        network.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                xs_val, ys_val = data
                xs_val = xs_val.to(device)
                ys_val = ys_val.to(device)

                y_preds_val = network(xs_val)
                y_preds_val = y_preds_val.squeeze(dim=1)

                loss_val = loss_fn(y_preds_val, ys_val)
                val_loss += loss_val.item()
        average_val_loss = val_loss / len(val_loader)
        print(f"Validation loss for epoch {epoch} is {average_val_loss}")
        writer.add_scalar('Loss/validate', average_val_loss, epoch)

        network.train()  # Set the model back to training mode


    

    print(f"run $ tensorboard --logdir={log_dir} \n to visualize logs")

