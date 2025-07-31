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
    parser.add_argument('--conf', default="config/rl_agent.ini")
    args = parser.parse_args()

    # Read and save config file
    conf = ConfigParser.read(args.conf)

    # Create logging directory
    if conf.model.pretrained and conf.model.do_resume:
        log_dir = conf.model.pretrained
        run_name = os.path.basename(log_dir)
        conf_path = os.path.join(log_dir, "settings.ini")
        conf = ConfigParser.read(conf_path)
    else:
        run_name = generate_run_name(args.fantasy_name)
        log_dir = os.path.join(conf.save.root, run_name)
        os.makedirs(log_dir)

    # Save config
    ConfigParser.save(conf, os.path.join(log_dir, "settings.ini"))

    # Create tb_handler 
    tb_handler = TensorboardHandler(log_dir)

    agent_handler = RLAgentHandler(conf, run_name, tb_handler)
    agent_handler.create()
    agent_handler.train()

    # Get best results
    episode, reward, spars, dmap, _ = agent_handler.get_best_results()

    # Log each flattened metric to both TensorBoard 
    flattened_metric_dict = {}
    flattened_metric_dict["episode"] = episode
    flattened_metric_dict["reward"] = reward
    flattened_metric_dict["spars"] = spars
    flattened_metric_dict["dmap"] = dmap
    
    tb_handler.log_hparams({
        'episodes': conf.train.episodes,
        'batch_size': conf.train.batch_size, 
        'optimizer': conf.model.actor_optimizer, 
        'start_lr': conf.model.actor_init_lr, 
        'weight_decay': conf.model.actor_weight_decay, 
        'entropy_coef': conf.model.actor_entropy_coef
    },
    metric_dict=flattened_metric_dict)