import argparse
from typing import List

import torch

import headmeta, metric


class DataModule:
    """Interface for custom data.

    This module handles datasets and is the class that you need to inherit
    from for your custom dataset. This class gives you all the handles so that
    you can train with a new `--dataset=mydataset`. The particular configuration
    of keypoints and skeleton is specified in the `headmeta` instances.
    """

    #: Data loader batch size.
    batch_size = 8

    #: Data loader number of workers.
    loader_workers = 0

    #: A list of head metas for this dataset.
    #: Set as instance variable (not class variable) in derived classes
    #: so that different instances of head metas are created for different
    #: instances of the data module. Head metas contain the base stride which
    #: might be different for different data module instances.
    #: When loading a checkpoint, entries in this list will be matched by
    #: name and dataset to entries in the checkpoint and overwritten here.
    head_metas: List[headmeta.Base] = None

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Commond line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def metrics(self) -> List[metric.Base]:
        """Define a list of metrics to be used for eval."""
        raise NotImplementedError

    def train_loader(self) -> torch.utils.data.DataLoader:
        """Loader of the training dataset."""
        raise NotImplementedError

    def val_loader(self) -> torch.utils.data.DataLoader:
        """Loader of the validation dataset.

        The augmentation and preprocessing should be the same as for train_loader.
        The only difference is the set of data. This allows to inspect the
        train/val curves for overfitting.

        As in the train_loader, the annotations should be encoded fields
        so that the loss function can be computed.
        """
        raise NotImplementedError

    def eval_loader(self) -> torch.utils.data.DataLoader:
        """Loader of the evluation dataset.

        For local runs, it is common that the validation dataset is also the
        evaluation dataset. This is then changed to test datasets (without
        ground truth) to produce predictions for submissions to a competition
        server that holds the private ground truth.

        This loader shouldn't have any data augmentation. The images should be
        as close as possible to the real application.
        The annotations should be the groud truth annotations similarly to
        what the output of the decoder is expected to be.
        """
        raise NotImplementedError
