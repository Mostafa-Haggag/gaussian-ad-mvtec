"""This file runs the main train/val loop, etc... using Lightning Trainer."""
import argparse
from importlib import import_module
import os
import shutil
import numpy as np

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .dataset import AnomalyDetectionDataset
import warnings


def get_dataset_class(name: str) -> AnomalyDetectionDataset:
    module = import_module("src.datasets." + name)
    # import_module comes from Python's importlib module and allows dynamic importing of a module by its string name.
    # This is especially useful when you need to import a module at runtime, where the module name is determined
    # programmatically.
    # It returns the imported module as an object,
    # which you can interact with as you would with a statically imported module.
    Dataset = module.DATASET # he calls MVTecAD in here
    # DATASET is variable inside the module
    if not issubclass(Dataset, AnomalyDetectionDataset):
        raise ValueError(
            "Dataset {} must inherit from "
            "src.common.dataset.AnomalyDetectionDataset".format(Dataset)
        )
    return Dataset


def get_model_class(name: str) -> LightningModule:
    module = import_module("src." + name)
    return module.MODEL


def new_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # i want to know which model you should be using in here
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The model folder to instantiate and train "
        "(relative to the src folder)",
    )
    # This is used to get to know the correct dataset that you should work on
    parser.add_argument(
        "--dataset",
        type=str,
        default="mvtecad",
        help="The dataset module to instantiate "
        "(relative to src.datasets folder)",
    )
    # the number of gpus that shoud be used
    parser.add_argument("--gpus", type=str, default=None)
    # what is the difference between number of iteration
    parser.add_argument(
        "--max_nb_iters",
        type=int,
        default=None,
        help="Maximum number of iterations to train for "
        "(last epoch still finishes",
    )
    parser.add_argument(
        "--min_nb_iters",
        type=int,
        default=None,
        help="Minimum number of iterations to train for "
        "(last epoch still finishes",
    )
    # and the number of epochs
    # what is the difference between both ?

    parser.add_argument(
        "--max_nb_epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--min_nb_epochs",
        type=int,
        default=None,
        help="Minimum number of epochs to train for",
    )
    # number of times to evalute
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1,
        help="Validate every --eval_freq epochs",
    )
    # path where logs should go
    parser.add_argument(
        "--logpath",
        type=str,
        default=os.getcwd(),
        help="The path where logs should go",
    )
    # what is this and how is it to be used?
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Use a deterministic version for logging",
    )

    # what is this exacly? we shoud understand
    parser.add_argument(
        "--fold",
        type=int,
        choices=range(5),
        help="The index of this training in a k-fold " "cross-validation",
        default=0,
    )
    return parser


def main(
    hparams: argparse.Namespace,
    Dataset: AnomalyDetectionDataset,
    Model: LightningModule,
) -> None:
    # init module (it will init the Dataset)
    model = Model(hparams, Dataset)

    if hparams.max_nb_iters is not None or hparams.min_nb_iters is not None:
        steps_per_epoch = len(model.train_dataloader())

    if hparams.max_nb_iters is not None:
        if hparams.max_nb_epochs is not None:
            raise RuntimeError(
                "don't specify both --max_nb_epochs and --max_nb_iters"
            )
        hparams.max_nb_epochs = np.ceil(
            hparams.max_nb_iters / steps_per_epoch
        ).astype(int)

    if hparams.min_nb_iters is not None:
        if hparams.min_nb_epochs is not None:
            raise RuntimeError(
                "don't specify both --min_nb_epochs and --min_nb_iters"
            )
        hparams.min_nb_epochs = np.ceil(
            hparams.min_nb_iters / steps_per_epoch
        ).astype(int)

    if hparams.max_nb_epochs is None:
        raise ValueError(
            "either max_nb_epochs or max_nb_iters need to be specified"
        )
    if hparams.min_nb_epochs is None:
        warnings.warn(
            "min_nb_epochs not specified, setting to 1", RuntimeWarning
        )
        hparams.min_nb_epochs = 1

    # Same as the default TensorBoardLogger in lightning, but with args.version.
    logger = TensorBoardLogger(
        save_dir=hparams.logpath,  # Path to save logs
        version=hparams.version,  # Version for the logs
        name="lightning_logs",  # Name of the log directory
    )
    if hparams.model == "classifier":
        # Only allow auroc-based validation in fully supervised classification.
        monitor = "auroc/val"
        mode = "max"
    else:
        monitor = "loss/val"
        mode = "min"

    # Construct the default ModelCheckpoint in order to adjust it.
    version_path = "{}/{}/version_{}".format(
        hparams.logpath, logger.experiment.name, logger.experiment.version
    )
    ckpt_path = "{}/{}".format(version_path, "checkpoints")

    early_stopping_callback = EarlyStopping(
        monitor=monitor, mode=mode, verbose=True, patience=3
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=True,
        monitor=monitor,
        mode=mode,
    )  # save best model

    trainer = Trainer(
        logger=logger,
        default_save_path=hparams.logpath,
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        min_nb_epochs=hparams.min_nb_epochs,
        check_val_every_n_epoch=hparams.eval_freq,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stopping_callback,
    )

    try:
        trainer.fit(model)
    except RuntimeError as e:
        if not ("corrupted" in str(e) or "in loading state_dict" in str(e)):
            raise  # Something else is wrong.
        print(e)
        # If the job is put back into queue when writing weights, the file will
        # be corrupted.
        # In this case, there is nothing we can do except to start fresh.
        print("Removing broken checkpoint: {}".format(version_path))
        shutil.rmtree(version_path, ignore_errors=True)
        # and restart training
        trainer.fit(model)

    trainer.test(model)  # re-load best model for testing ;)


if __name__ == "__main__":
    # First load a dummy parser to instantiate the correct dataset and model.
    parser = new_parser()
    parser.add_argument("args", nargs=argparse.REMAINDER)
    # This line is adding an argument named "args" to the argument parser
    # nargs=argparse.REMAINDER means that this argument will capture all remaining command-line arguments after the
    # recognized ones. Essentially, this collects any arguments not explicitly defined before this in the argparse parser.
    globalparams, unknown = parser.parse_known_args()
    # globalparams will contain the recognized (known) arguments (those explicitly defined in the add_argument() calls).
    # unknown will store any unknown arguments (those that do not match any add_argument()).
    # Together, this setup allows you to handle both predefined and unrecognized arguments.
    Dataset = get_dataset_class(globalparams.dataset)
    Model = get_model_class(globalparams.model)

    parser = new_parser()
    # Give the module and the dataset a chance to add its own params.
    parser = Dataset.add_dataset_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    main(hparams, Dataset, Model)
