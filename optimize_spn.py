import argparse
import os
import optuna

from utils.config_parser import ConfigParser
from state_predictor.dataloader import create_pruning_dataloader
from src.model.spn_handler import SPNHandler
from utils.tensorboard_handler import TensorboardHandler
from train_spn import generate_run_name


def objective(trial):
    # Suggest hyperparameters
    epochs = trial.suggest_int('epochs', 200, 2000, step=100)
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512, 1024, 2048, 4096, 8192, 16384])

    optimizer_type = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'adamw'])
    start_lr = trial.suggest_categorical(
        'start_lr', [c * 10**-e for e in range(2, 7) for c in range(1, 10)]
    )
    weight_decay = trial.suggest_categorical(
        'weight_decay', [c * 10**-e for e in range(2, 7) for c in range(1, 10)]
    )
    momentum = trial.suggest_uniform('momentum', 0.5, 0.99) if optimizer_type == 'sgd' else None
    #spars_loss_weight = trial.suggest_float('spars_loss_weight', 0.1, 1.0, step=0.1)
    dmap_loss_weight = trial.suggest_categorical('dmap_loss_weight', [0.1, 1.0, 10.0, 100.0])


    # Read config and set hyperparameters
    conf = ConfigParser.read("config/spn.ini")
    conf.train.epochs = epochs
    conf.train.batch_size = batch_size
    conf.model.optimizer = optimizer_type
    conf.model.start_lr = start_lr
    conf.model.weight_decay = weight_decay
    if momentum is not None:
        conf.model.momentum = momentum

    # Create logging directory
    run_name =  generate_run_name("optuna")
    log_dir = os.path.join(conf.save.root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    ConfigParser.save(conf, os.path.join(log_dir, 'settings.ini'))

    # Tensorboard handler
    tb_handler = TensorboardHandler(log_dir)

    # Create dataloaders
    train_dataloader = create_pruning_dataloader(conf, split_type='training')
    val_dataloader = create_pruning_dataloader(conf, split_type='validation')

    # Load and train model
    spn_handler = SPNHandler(conf, log_dir, tb_handler)
    spn_handler.create()
    spn_handler.train(train_dataloader, val_dataloader)

    # Validation
    val_loss, val_metrics = spn_handler.evaluate(val_dataloader)

    # Log each flattened metric to both TensorBoard and Optuna
    flattened_metric_dict = {}
    flattened_metric_dict["val_loss"] = val_loss
    for key, subdict in val_metrics.items():
            for metric_name, value in subdict.items():
                metric_full_name = f"{key}.{metric_name}"
                flattened_metric_dict[metric_full_name] = value
    
    tb_handler.log_hparams({
        'epochs': epochs,
        'batch_size': batch_size, 
        'optimizer': optimizer_type, 
        'start_lr': start_lr, 
        'weight_decay': weight_decay, 
        'momentum': momentum, 
        #'spars_loss_weight': spars_loss_weight,
        'dmap_loss_weight': dmap_loss_weight
    },
    metric_dict=flattened_metric_dict)




    return val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=500, help='Number of optimization trials')
    args = parser.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.trials)

    print(f'Best trial: {study.best_trial.number}')
    print(f'Best value (val_loss): {study.best_value}')
    print('Best hyperparameters:', study.best_params)
