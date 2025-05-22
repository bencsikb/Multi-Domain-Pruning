import os 
import argparse

from utils.config_parser import ConfigParser
from src.model.rl_agent_handler import RLAgentHandler
from utils.tensorboard_handler import TensorboardHandler
from utils.common_utils import generate_run_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fantasy_name', type=str) 
    parser.add_argument('--device', default='')
    args = parser.parse_args()

    # Read and save config file
    conf = ConfigParser.read("config/rl_agent.ini")

    # Create logging directory
    run_name = generate_run_name(args.fantasy_name)
    log_dir = os.path.join(conf.save.root, run_name)
    os.makedirs(log_dir)

    # Save config
    #ConfigParser.save(conf, os.path.join(log_dir, "settings.ini"))

    # Create tb_handler 
    tb_handler = TensorboardHandler(log_dir)

    agent_handler = RLAgentHandler(conf, run_name, tb_handler)
    agent_handler.create()
    agent_handler.train()

    # load pretrained nets: 
        # - net for pruning --> existing model_handler
        # - SPN net --> model_handler?