import os 

from utils.config_parser import ConfigParser
from src.model.spn_handler import SPNHandler
from state_predictor.dataloader import create_pruning_dataloader

if __name__ == "__main__":
    pass

    # argument parser

    # Read and save config file
    conf = ConfigParser.read("config/spn.ini")
    # ConfigParser.save(conf, os.path.join(conf.samples.save_path, "settings.ini"))

    # create loggers (rl, tb, txt)

    # create dataloaders
    train_dataloader = create_pruning_dataloader(conf, split_type="train")
    # val_dataloader = ..

    # load or define SPN model
    spn_handler = SPNHandler(conf)
    spn_handler.create()
    spn_handler.train(train_dataloader, train_dataloader)

    # Train SPN
    