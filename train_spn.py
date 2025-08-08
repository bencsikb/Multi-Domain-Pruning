import os 
import argparse
import datetime
import uuid

from utils.config_parser import ConfigParser
from src.model.spn_handler import SPNHandler
from utils.tensorboard_handler import TensorboardHandler
from state_predictor.dataloader import create_pruning_dataloader

def generate_run_name(prefix=""):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    name = f"{date_str}_{short_hash}_{prefix}" if prefix else f"{date_str}_{short_hash}"
    return name

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fantasy_name', type=str) 
    parser.add_argument('--device', default='')
    parser.add_argument('--conf', default="config/spn_transformer.ini")
    args = parser.parse_args()

    # Read and save config file
    conf = ConfigParser.read(args.conf)

    # Create logging directory
    run_name = generate_run_name(args.fantasy_name)
    log_dir = os.path.join(conf.save.root, run_name)
    os.makedirs(log_dir)

    # Save config
    ConfigParser.save(conf, os.path.join(log_dir, "settings.ini"))

    # Create tb_handler 
    tb_handler = TensorboardHandler(log_dir)

    # create dataloaders
    train_dataloader = create_pruning_dataloader(conf, split_type="training")
    val_dataloader = create_pruning_dataloader(conf, split_type="validation")

    # load or define SPN model
    spn_handler = SPNHandler(conf, run_name, tb_handler)
    spn_handler.create()
    spn_handler.train(train_dataloader, val_dataloader)

    