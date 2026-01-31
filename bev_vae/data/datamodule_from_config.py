from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DataModuleFromConfig(LightningDataModule):
    def __init__(
        self, 
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        test_batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        ):
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        if self.hparams.train is not None:
            print(f'Train dataset has {len(self.hparams.train)} samples')
        if self.hparams.val is not None:
            print(f'Val dataset has {len(self.hparams.val)} samples')
        if self.hparams.test is not None:
            print(f'Test dataset has {len(self.hparams.test)} samples')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.hparams.train, 
            batch_size=self.hparams.train_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=self.hparams.shuffle, 
            drop_last=self.hparams.drop_last)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.hparams.val, 
            batch_size=self.hparams.val_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            dataset=self.hparams.test, 
            batch_size=self.hparams.test_batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory, 
            shuffle=False)
