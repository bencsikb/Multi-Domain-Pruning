import os
import torch
from typing import List

from utils.tensorboard_handler import TensorboardHandler
from src.model.yolo_handler import YoloHandler
from src.model.spn_handler import SPNHandler
from src.training_components import get_optimizer, get_lr_scheduler
from pruning.model_pruner.stepwise_rl_pruner import StepWiseRLPruner
from pruning.channel_selection.channel_selector import ChannelSelector
from state_predictor.coder import Coder
from reinforcement_learning.model import actorNet, criticNet
from reinforcement_learning.losses import ActorLoss, CriticLoss
from reinforcement_learning.rewards import reward_function_proposed, reward_function_purl, reward_function_amc


class RLAgentHandler():
    def __init__(self, conf, run_name: str, tb_handler: TensorboardHandler) -> None:
        
        self._conf = conf
        self._run_name = run_name
        self._tb_handler = tb_handler
        self._log_dir_path = os.path.join(conf.save.root, run_name)

        self._spn_handler = self._load_spn()
        self._yolo_handler = self._load_yolo()
        self._model_pruner = self._get_model_pruner()
        self._coder = self._initialize_coder()

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
    
    def _get_model_pruner(self) -> StepWiseRLPruner:
        yolo_conf = self._conf.yolo    
        channel_selector = ChannelSelector(self._conf.channel_selection)  # TODO conf should be loaded from folder
        return StepWiseRLPruner(self._yolo_handler, yolo_conf, channel_selector)

    def _initialize_coder(self):
        """Initialize the coder with an example file.
           Should be called after initializing the StepWiseRLPruner, because of the model state.
        """
        
        state_example_df = self._model_pruner.data
        alpha_range = self._conf.alpha.min_max_steps[:2] #TODO
        return Coder(state_example_df, None, alpha_range)
    
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

    def _load_checkpoint(self):
        pass

    
    
    def create(self):
        
        state_shape = self._n_prunable_layers * len(self._state_features)

        self._actor_model = actorNet(state_shape, len(self._possible_alphas))
        self._actor_optimizer = get_optimizer(type = self._conf.actor_optimizer,
                                              model = self._actor_model,
                                              lr = self._conf.actor_init_lr,
                                              weight_decay = self._conf.actor_weight_decay                                                                       
                                            )  
        self._actor_loss = ActorLoss()

        self._critic_model = criticNet(state_shape, 1)
        self._critic_optimizer = get_optimizer(type = self._conf.critic_optimizer,
                                              model = self._critic_model,
                                              lr = self._conf.critic_init_lr,
                                              weight_decay = self._conf.critic_weight_decay                                                                       
                                            )          
        self._critic_loss = CriticLoss()

        self._lr_scheduler = get_lr_scheduler(type = self._conf.lr_scheduler,
                                              epochs = self._conf.train.epochs,
                                              optimizer = self._actor_optimizer)  
        
        self._episode = 0
        # TODO extend with loading pretrained Actor, Critic


    def train(self):

        while self.episode < self._conf.train.episodes:
            
            # === 2. Initialize Environment ===

            network_seq = []
            init_param_nmb = sum([param.nelement() for param in net_for_pruning.parameters()])

            action_seq = torch.full([conf.train.batch_size, 1, conf.prune.n_prunable_layers], -1.0)
            state_seq = torch.full([conf.train.batch_size, n_features, conf.prune.n_prunable_layers], -1.0)

            actions = []
            states = []
            rewards = []
            rewards_list = []
            values = []
            policies = []
            log_probs = []
            entropies = []
            errors = []
            dEs, dSs = [], []

            layer_cnt = 0
            sparsity_prev = torch.full([conf.train.batch_size], -1.0)
            dmap_prev = torch.full([conf.train.batch_size], -1.0)
            sparsity_prev = torch.full([conf.train.batch_size], -1.0)
            dmap_prev = torch.full([conf.train.batch_size], -1.0)

            self._model_pruner.reset_model_and_state()
            
            # === 3. Iterate Over Layers ===
            for i, layer in enumerate(self._yolo_handler.prunable_layers):

                # --- 3b. Get / Update State ---
                self._model_pruner.increment_layer()
                self._model_pruner.update_state()

                data = self._model_pruner.data # alpha + state
                encoded_data = self._coder.encode_state(data)
                state = ... # remove alpha from data
                encoded_state = ...

                # --- 3c. Actor Forward ---              

                probs, action_dist, log_softmax = self._actor_model(encoded_data)
                action = action_dist.sample()  # alpha index
                
                # --- 3d. Sample Action & Compute Info ---

                # entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(action).unsqueeze(1)
                policy = probs.gather(-1, action.unsqueeze(0))
                entropy = - (probs * log_softmax).sum(1, keepdim=True)

                for i in range(self._conf.train.batch_size):
                    action_seq[i, :, layer_cnt] = self._possible_alphas[action[i]] # not normalized.
                
                # --- 3e. Log Layer Info ---
                # --- 3f. Save Actions ---

                # --- 3g. Predict Error & Sparsity --
                
                spn_input_data = torch.cat((action_seq, state_seq[:, -1, :].unsqueeze(1)), dim=1).view(
                        [conf.train.batch_size, -1]).type(torch.float32).to(device)
                
                prediction = self._spn_handler.predict(spn_input_data)

                spars, dmap = prediction[:,0], prediction[:,1]
                
                # --- 3h. Compute Reward ---
                reward = self._get_reward(self._conf.reward.type, spars, dmap)

                # --- 3i. Save Trajectory Step ---

                # --- 3j. Update for Next Layer ---

            # === 4. Optional Evaluation ===

            # === 5. Select Best Result ===

            # === 6. Compute Returns ===

            # === 7. Prepare Log Probs ===

            # === 8. Compute Losses ===

            # === 9. Backpropagation ===

            # === 10. Logging ===

            # === 11. Save Checkpoint ===

            # === 12. End of Episode ===




    def _get_reward(self, reward_type, spars, dmap):

        if reward_type == 'proposed':
            reward = reward_function_proposed(self._coder.decode_label(spars),
                                              self._conf.reward.target_spars,
                                              self._coder.decode_label(dmap), 
                                              self._conf.reward.target_dmap,
                                              self._conf.reward.spars_coef,
                                              self._conf.reward.dmap_coef,
                                              self._conf.reward.beta )
        elif reward_type == 'purl':
            reward = reward_function_purl(self._coder.decode_label(spars),
                                            self._conf.reward.target_spars,
                                            self._coder.decode_label(dmap), 
                                            self._conf.reward.target_map,
                                            self._conf.reward.map_before, # ? TODO 
                                            self._conf.reward.beta )             

        elif reward_type == 'amc':
            reward = reward_function_amc(self._coder.decode_label(spars),
                                         self._coder.decode_label(dmap))
                                         # init_params TODO

        return reward.unsqueeze(1)
    

    def _init_state_sequence(self):
        pass

    def _update_state_sequence(self, state):
        
        i = layer_cnt

        state[:, 0, i] = normalize(layer.in_channels, 0, 1024)
        state[:, 1, i] = normalize(layer.out_channels, 0, 1024)
        state[:, 2, i] = normalize(layer.kernel_size[0], 0, 3)
        state[:, 3, i] = normalize(layer.stride[0], 0, 2)
        state[:, 4, i] = normalize(layer.padding[0], 0, 1)
        state[:, 5, i] = sparsity
        if state.shape[1] == 7:
            state[:, 6, i] = dmap

        return state

    def _init_action_sequence(self):
        pass

    def _update_action_sequence(self):
        pass



