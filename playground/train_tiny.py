from tiny_network import TinyNN
from data_utils import get_device, get_input_output, get_tb_logging_dir, seed_experiment
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

    
if __name__ == "__main__":
    
    seed_experiment()

    # write tb logs, view via 
    log_dir = get_tb_logging_dir("Vanilla-Pytorch")
    writer = SummaryWriter(log_dir=log_dir)

    device = get_device()
    print(f"Using {device} device")

    xs, ys = get_input_output()
    xs = xs.to(device)
    ys = ys.to(device)

    network = TinyNN().to(device)

    loss_fn = nn.MSELoss()

    # learning rate (TODO hyperparameter to be tuned by hand)
    lr = 1e-3 
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    for epoch in range(0, 1000):
        
        print(f"Epoch {epoch}")

        # 1 a. forward pass
        y_preds = network(xs)

        # match shapes of predicted `ys` to ground truth `ys`
        # squeeze from torch.Size([6, 1]) to torch.Size([6]) 
        y_preds = y_preds.squeeze(dim=1) 

        # 1 b.  calculate loss
        loss = loss_fn(y_preds, ys)

        # 2. backward pass (Calculate grads)
        loss.backward()

        # 3. Update params (optimizer step)
        optimizer.step()

        # 4. zero grad before epoch
        optimizer.zero_grad()

        print(f"Loss for epoch {epoch} is {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), epoch)
    
    print(f"run $ tensorboard --logdir={log_dir} \n to visualize logs")
