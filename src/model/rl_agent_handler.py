import os
from typing import List

from utils.tensorboard_handler import TensorboardHandler
from src.model.yolo_handler import YoloHandler
from src.model.spn_handler import SPNHandler


class RLAgent():
    def __init__(self, conf, run_name: str, tb_handler: TensorboardHandler) -> None:
        
        self._conf = conf
        self._run_name = run_name
        self._tb_handler = tb_handler
        self._log_dir_path = os.path.join(conf.save.root, run_name)

        self._spn_handler = self._load_spn()
        self._yolo_handler = self._load_yolo()

        self._possible_alphas = self._get_alphas()
        self._n_prunable_layers = self._get_n_prunable_layers()

        self._state_features = self._conf.state_features


    def _load_spn(self) -> SPNHandler:
        
        run_path = self._conf.spn.root        
        run_name = os.path.basename(run_path)
        conf_path = os.path.join(run_path, "settings.ini")
        # load or define SPN model
        spn_handler = SPNHandler(conf_path, run_name=run_name)
        spn_handler.create(is_pretrained=True)

        return spn_handler
    
    
    def _load_yolo(self) -> YoloHandler:

        yolo_conf = self._conf.yolo
        yolo_handler = YoloHandler(yolo_conf)

        return yolo_handler   

    def _get_alphas(self) -> List:
        """
        Get the possible alpha values from SPN data generation settings.ini 
        """     
        from utils.config_parser import ConfigParser
        from pruning.channel_selection.alpha_functions import ActionFunc

        data_path = self._conf.spn.data_path        
        conf_path = os.path.join(data_path, "settings.ini")
        spn_data_conf =  ConfigParser.read(conf_path)

        action_generator = ActionFunc(spn_data_conf)

        return action_generator.alphas
    
    def _get_n_prunable_layers(self) -> int:
        """
        Get the number of prunable layers from the yolo_handler.
        """

        self._yolo_handler.determine_prunable_layers()
        return self._yolo_handler.n_prunable_layers

    
    
    def create(self):

        from reinforcement_learning.model import actorNet, criticNet
        from src.training_components import get_loss_function, get_optimizer, get_lr_scheduler
        
        state_shape = self._n_prunable_layers * len(self._state_features)

        self._actor_model = actorNet(state_shape, len(self._possible_alphas))
        self._actor_optimizer = get_optimizer(type = self._conf.actor_optimizer,
                                              model = self._actor_model,
                                              lr = self._conf.actor_init_lr,
                                              weight_decay = self._conf.actor_weight_decay                                                                       
                                            )  
        self._actor_loss = ...

        self._critic_model = criticNet(state_shape, 1)
        self._critic_optimizer = get_optimizer(type = self._conf.critic_optimizer,
                                              model = self._critic_model,
                                              lr = self._conf.critic_init_lr,
                                              weight_decay = self._conf.critic_weight_decay                                                                       
                                            )          
        self._critic_loss = ...

        self._lr_scheduler = get_lr_scheduler(type = self._conf.lr_scheduler,
                                              epochs = self._conf.train.epochs,
                                              optimizer = self._actor_optimizer)  




