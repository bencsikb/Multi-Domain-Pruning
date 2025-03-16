import os 

from utils.config_parser import ConfigParser
from src.model.spn_handler import SPNHandler

if __name__ == "__main__":
    pass

    # argument parser

    # Read and save config file
    conf = ConfigParser.read("config/spn.ini")
    # ConfigParser.save(conf, os.path.join(conf.samples.save_path, "settings.ini"))

    # create loggers (rl, tb, txt)

    # create dataloaders

    # load or define SPN model
    spn_handler = SPNHandler(conf)

    # Train SPN
    