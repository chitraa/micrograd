from data_utils import TinyDataset, seed_experiment, get_tb_logging_dir, random_math_function
import torch
import os
import pytorch_lightning as pl 
from tiny_network_lightning import TinyNNLightning, get_model_checkpoint_callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger



def train(log_dir, experiment_name):

    # Set the seed for reproducibility

    seed = seed_experiment()
    pl.seed_everything(seed)

    tiny_dataset = TinyDataset()

    train_dataset, val_dataset = torch.utils.data.random_split(
        tiny_dataset, [0.6, 0.4])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=False)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False)
    
    network = TinyNNLightning()
    
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)
    
    # custom ModelCheckpoint callback to save model
    exp_dir = os.path.join(log_dir, experiment_name)
    checkpoint_callback = get_model_checkpoint_callback(exp_dir)

    # custom early stopping callback, in this experiment it stopped at epoch 250 when 
    # set to max_epochs=1000
    early_stop_callback = EarlyStopping(monitor="epoch_val_loss", 
                                        min_delta=0.001, patience=3, 
                                        verbose=False, mode="min")

    
    trainer = pl.Trainer(max_epochs=1000, logger=tensorboard_logger, 
                         callbacks=[checkpoint_callback, early_stop_callback])
    

    trainer.fit(network, train_loader, val_loader)

    print("Best model: ", checkpoint_callback.best_model_path)

    logged_metrics = trainer.callback_metrics
    print(logged_metrics)

    print(f"run $ tensorboard --logdir={log_dir} \n to visualize logs")

def evaluate_model(model, input_values):
    model.eval()  # Set the model to evaluation mode
    model_outputs = []

    with torch.no_grad():
        for input_value in input_values:
            # Convert the scalar input to a 3-dimensional input array
            input_array = torch.tensor([input_value, input_value, input_value], dtype=torch.float32)
            input_array = input_array.unsqueeze(0)  # Add batch dimension

            # Forward pass through the model
            model_output = model(input_array).item()
            model_outputs.append(model_output)

            # Evaluate the random mathematical function for comparison
            true_output = random_math_function(input_array.squeeze().numpy())
            
            # Print the results for the current input
            print(f"Input: {input_array.squeeze().numpy()}, Model Output: {model_output}, True Output: {true_output}")

    return model_outputs

def test(log_dir, experiment_name, checkpoint_filename):
    exp_dir = os.path.join(log_dir, experiment_name)
    ckpt_path = os.path.join(exp_dir, checkpoint_filename)
    

    # Instantiate a PyTorch Lightning Trainer
    trainer = pl.Trainer()

    loaded_model = TinyNNLightning.load_from_checkpoint(ckpt_path)

    # Access hyperparameters from the loaded model
    loaded_hyperparameters = loaded_model.hparams


    print(loaded_hyperparameters)

    input_values = [-2, -1, 0, 1, 2, 3, 4]
    model_outputs = evaluate_model(loaded_model, input_values)



if __name__ == "__main__":
    logging_dir_name = "Pytorch-Lightning"
    experiment_name = "mlp41-epoch1000-early-stop-true"
    best_model_ckpt = "epoch=249-step=500.ckpt"
       # write tb logs, view via 
    log_dir = get_tb_logging_dir(logging_dir_name)

    
    # train(log_dir, experiment_name)
    test(log_dir, experiment_name, best_model_ckpt)