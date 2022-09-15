from pytorch_lightning import LightningDataModule
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import ShaDowKHopSampler


class InductiveNodeLoader(LightningDataModule):
    def __init__(self, data, num_neighbours, batch_size, num_workers):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neightbours = num_neighbours

    def train_dataloader(self):
        data = self.data.subgraph(self.data.train_mask)
        return NeighborLoader(data=data,
                              num_neighbors=self.num_neightbours,
                              input_nodes=data.train_mask,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers)

    def val_dataloader(self):
        data = self.data.subgraph(~self.data.test_mask)
        return NeighborLoader(data=data,
                              num_neighbors=self.num_neightbours,
                              input_nodes=data.val_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)

    def test_dataloader(self):
        return NeighborLoader(data=self.data,
                              num_neighbors=self.num_neightbours,
                              input_nodes=self.data.test_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)

    def predict_dataloader(self):
        return NeighborLoader(data=self.data,
                              num_neighbors=self.num_neightbours,
                              input_nodes=self.data.test_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)


class InductiveShadowLoader(LightningDataModule):
    def __init__(self, data, depth, num_neighbours, batch_size, num_workers):
        super().__init__()
        self.data = data
        self.depth = depth
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_neightbours = num_neighbours

    def train_dataloader(self):
        data = self.data.subgraph(self.data.train_mask)
        return ShaDowKHopSampler(data=data, depth=self.depth,
                              num_neighbors=self.num_neightbours,
                              node_idx=data.train_mask,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers)

    def val_dataloader(self):
        data = self.data.subgraph(~self.data.test_mask)
        return ShaDowKHopSampler(data=data, depth=self.depth,
                              num_neighbors=self.num_neightbours,
                              node_idx=data.val_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)

    def test_dataloader(self):
        return ShaDowKHopSampler(data=self.data, depth=self.depth,
                              num_neighbors=self.num_neightbours,
                              node_idx=self.data.test_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)


    def predict_dataloader(self):
        return ShaDowKHopSampler(data=self.data, depth=self.depth,
                              num_neighbors=self.num_neightbours,
                              node_idx=self.data.test_mask,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=self.num_workers)




