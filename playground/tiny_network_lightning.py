import os
import torch
from torch import nn
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint

def get_model_checkpoint_callback(exp_dir):
    
        # custom ModelCheckpoint callback to save model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        auto_insert_metric_name=True,
        save_top_k=2,
        every_n_epochs=10,
        dirpath=exp_dir
    )
    return checkpoint_callback


class TinyNNLightning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        """ Layers(network) definition
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.training_step_outputs = []
        self.validation_step_outputs = []
        # self.default_save_path = "/path/to/save/hparams"
        self.save_hyperparameters(ignore=['training_step_outputs', 'validation_step_outputs'])
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(3, 4),
            nn.Tanh(),
            nn.Linear(4, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Tanh()
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    
    # extra in lightning
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def _get_batch_loss(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze(dim=1)
        return self.loss(y_hat, y)
    

    # extra in lightning
    def training_step(self, batch, batch_idx):
        loss = self._get_batch_loss(batch)

        self.log('train_loss', loss)
        self.training_step_outputs.append({"train_loss": loss})
        return loss

    # extra in lightning
    def validation_step(self, batch, batch_idx):
        loss = self._get_batch_loss(batch)

        self.log('val_loss', loss)
        self.validation_step_outputs.append({"val_loss": loss})
        return loss
    

    def on_train_epoch_end(self):
        # hooks, called at the end of epoch
        # https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        avg_loss = torch.stack([x['train_loss'] for x in self.training_step_outputs]).mean()
        self.log('epoch_train_loss', avg_loss)
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # hooks, called at the end of epoch
        # https://github.com/Lightning-AI/pytorch-lightning/pull/16520
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        self.log('epoch_val_loss', avg_loss)
        self.validation_step_outputs.clear() 