import os 
import argparse

from utils.config_parser import ConfigParser
from src.model.spn_handler import SPNHandler
from state_predictor.dataloader import create_pruning_dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str) #TODO make it optional
    parser.add_argument('--device', default='')
    args = parser.parse_args()

    # Read and save config file
    conf = ConfigParser.read("config/spn.ini")

    # Create logging directory
    log_dir = os.path.join(conf.save.root, args.name)

    try:
        if os.path.exists(log_dir):
            raise FileExistsError(f"Folder already exists: {log_dir}")
        os.makedirs(log_dir)
    except FileExistsError as e:
        print(e)


    # Save config
    ConfigParser.save(conf, os.path.join(log_dir, "settings.ini"))

    # create dataloaders
    train_dataloader = create_pruning_dataloader(conf, split_type="training")
    val_dataloader = create_pruning_dataloader(conf, split_type="validation")

    # load or define SPN model
    spn_handler = SPNHandler(conf, log_dir)
    spn_handler.create()
    spn_handler.train(train_dataloader, val_dataloader)

    