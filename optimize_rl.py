import argparse
import os
import optuna

from utils.config_parser import ConfigParser
from src.model.rl_agent_handler import RLAgentHandler
from utils.tensorboard_handler import TensorboardHandler
from utils.common_utils import generate_run_name


def objective(trial, config, obj_metric):
    # Suggest hyperparameters
    episodes = trial.suggest_int('episodes', 200, 400, step=100)
    batch_size = trial.suggest_categorical('batch_size', [1024])

    optimizer_type = trial.suggest_categorical('optimizer', ['adam']) #,, 'adamw 'lamb'])
    start_lr = trial.suggest_categorical(
        'start_lr', [c * 10**-e for e in range(2,4) for c in range(1, 10)]
    )
    weight_decay = trial.suggest_categorical(
        'weight_decay', [c * 10**-e for e in range(4, 6) for c in range(1, 10)]
    )
    entropy_coef = trial.suggest_categorical(
        'entropy_coef', [c * 10**-e for e in range(4, 6) for c in range(1, 10)]
    )
    entropy_factor = trial.suggest_categorical('entropy_factor', [ e for e in range(10, 150, 10)])
    spars_coeff = trial.suggest_categorical('spars_coeff', [0.1, 0.2, 0.3])
    dmap_coeff = 1 - spars_coeff


    # Read config and set hyperparameters
    conf = ConfigParser.read(config)
    conf.train.episodes = episodes
    conf.train.batch_size = batch_size
    conf.model.actor_optimizer = optimizer_type
    conf.model.critic_optimizer = optimizer_type
    conf.model.actor_init_lr = start_lr
    conf.model.critic_init_lr = start_lr
    conf.model.actor_weight_decay = weight_decay
    conf.model.critic_weight_decay = weight_decay
    conf.model.actor_entropy_coef = entropy_coef
    conf.model.entropy_factor = entropy_factor
    conf.reward.spars_coeff = spars_coeff
    conf.reward.dmap_coeff = dmap_coeff

    # Create logging directory
    run_name =  generate_run_name("optuna")
    log_dir = os.path.join(conf.save.root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    ConfigParser.save(conf, os.path.join(log_dir, 'settings.ini'))

    # Tensorboard handler
    tb_handler = TensorboardHandler(log_dir)

    # Load and train model
    rl_handler = RLAgentHandler(conf, run_name, tb_handler)
    rl_handler.create()
    rl_handler.train()

    # Get best results
    episode, reward, spars, dmap, _ = rl_handler.get_best_results()

    # Log each flattened metric to both TensorBoard and Optuna
    flattened_metric_dict = {}
    flattened_metric_dict["episode"] = episode
    flattened_metric_dict["reward"] = reward
    flattened_metric_dict["spars"] = spars
    flattened_metric_dict["dmap"] = dmap
    
    tb_handler.log_hparams({
        'episodes': episodes,
        'batch_size': batch_size, 
        'optimizer': optimizer_type, 
        'start_lr': start_lr, 
        'weight_decay': weight_decay, 
        'entropy_coef': entropy_coef,
        'entropy_factor': entropy_factor,
        'dmap_coeff': dmap_coeff
    },
    metric_dict=flattened_metric_dict)


    return - flattened_metric_dict[obj_metric] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200, help='Number of optimization trials')
    parser.add_argument('--config', type=str, default="config/rl_agent.ini")
    parser.add_argument('--obj_metric', type=str, default="reward")

    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, config=args.config, obj_metric=args.obj_metric), n_trials=args.trials)

    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value (val_loss): {study.best_value}')
    print('Best hyperparameters:', study.best_params)
