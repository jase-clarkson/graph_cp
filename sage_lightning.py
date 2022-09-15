import os.path as osp
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torchmetrics import Accuracy

from torch_geometric import seed_everything
from torch_geometric.datasets import Flickr, Reddit2
from torch_geometric.nn import GraphSAGE
from inductive_datamodule import InductiveNodeLoader
import pickle as pkl


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, aggr='mean'):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers,
                             out_channels, dropout=dropout,
                             norm=BatchNorm1d(hidden_channels), aggr=aggr)

        self.num_layers = num_layers
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=y_hat.size(0))

    def predict_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        return y_hat.softmax(dim=-1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


def main():
    seed_everything(42)

    dataset = 'Flickr'
    time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    fstr = time + '_' + dataset
    exp_path = os.path.join('experiments', fstr)
    os.mkdir(exp_path)

    data_path = os.path.join('data', dataset)
    dataset = eval(dataset)(data_path)
    # dataset = Reddit2('data/Reddit2')
    data = dataset[0]
    datamodule = InductiveNodeLoader(data,
                                   num_neighbours=[25, 10],
                                   batch_size=512,
                                   num_workers=8)

    model = Model(dataset.num_node_features,
                  dataset.num_classes,
                  aggr='max')

    devices = torch.cuda.device_count()
    checkpoint = pl.callbacks.ModelCheckpoint(dirpath=exp_path,
                                              filename='{epoch}-{val_acc:.2f}',
                                              monitor='val_acc',
                                              mode='max',
                                              save_top_k=1)
    trainer = pl.Trainer(default_root_dir=exp_path,
                         accelerator='gpu',
                         devices=devices,
                         max_epochs=30,
                         callbacks=[checkpoint])

    trainer.fit(model, datamodule)
    print(f'{checkpoint.best_model_path}')
    trainer.test(ckpt_path='best', datamodule=datamodule)
    preds = trainer.predict(model, datamodule)
    preds = torch.cat(preds).numpy()
    print('Saving predictions')
    preds_save_path = os.path.join(exp_path, 'preds.pkl')
    with open(preds_save_path, 'wb') as f:
        pkl.dump(preds, f)

if __name__ == '__main__':
    main()