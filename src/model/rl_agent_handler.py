import os
import torch
from typing import List
from types import SimpleNamespace

from utils.tensorboard_handler import TensorboardHandler
from src.model.yolo_handler import YoloHandler
from src.model.spn_handler import SPNHandler
from utils.config_parser import ConfigParser
from src.training_components import get_optimizer, get_lr_scheduler
#from pruning.model_pruner.stepwise_rl_pruner import StepWiseRLPruner
from pruning.model_pruner.stepwise_pruner import StepWisePruner
from pruning.channel_selection.channel_selector import ChannelSelector
from state_predictor.coder import Coder
from reinforcement_learning.model import actorNet, criticNet
from reinforcement_learning.losses import ActorLoss, CriticLoss
from reinforcement_learning.rewards import reward_function_proposed, reward_function_purl, reward_function_amc


class RLAgentHandler():
    def __init__(self, conf, run_name: str, tb_handler: TensorboardHandler) -> None:
        
        self._conf = conf
        self._device = self._conf.train.device
        self._run_name = run_name
        self._tb_handler = tb_handler
        self._log_dir_path = os.path.join(conf.save.root, run_name)

        # Load configs
        self._conf = conf
        self._spn_conf = self._get_spn_config()
        self._samples_conf = self._get_samples_config()

        self._spn_handler = self._load_spn()
        self._yolo_handler = self._load_yolo()
        self._model_pruner = self._get_model_pruner()
        self._coder = self._initialize_coder()

        self._possible_alphas = self._get_alphas()
        self._n_prunable_layers = self._get_n_prunable_layers()

        self._state_features = self._spn_conf.model.state_features
    
    def _get_spn_config(self) -> SimpleNamespace:

        run_path = self._conf.spn.root        
        conf_path = os.path.join(run_path, "settings.ini")
        conf = ConfigParser.read(conf_path)
        return conf

    def _get_samples_config(self) -> SimpleNamespace:
        samples_conf_path = self._conf.yolo.samples_conf_path
        samples_conf = ConfigParser.read(samples_conf_path)
        return samples_conf

    def _load_spn(self) -> SPNHandler:

        run_path = self._conf.spn.root        
        run_name = os.path.basename(run_path)      
        spn_handler = SPNHandler(self._spn_conf, run_name=run_name)
        spn_handler.create(is_pretrained=True)
        return spn_handler
    
    
    def _load_yolo(self) -> YoloHandler:

        yolo_conf = self._samples_conf.model
        yolo_handler = YoloHandler(yolo_conf)

        return yolo_handler   
    
    def _get_model_pruner(self) -> StepWisePruner:
        """ Needed for loading the initial model state. """

        channel_selector = ChannelSelector(self._samples_conf.channel_selection) 

        pruner = StepWisePruner(model_handler=self._yolo_handler,
                                sample_handler=None,
                                conf = self._samples_conf,
                                channel_selector=channel_selector,
                                is_rl = True)

        return pruner

    def _initialize_coder(self):
        """Initialize the coder with an example file.
           Should be called after initializing the StepWiseRLPruner, because of the model state.
        """
        
        state_example_df = self._model_pruner.data
        alpha_range = self._samples_conf.alpha.min_max_steps[:2] #TODO
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

        state_feature_dim = self._n_prunable_layers * (len(self._state_features) - 1) # alpha is not needed -> -1

        self._actor_model = actorNet(state_feature_dim, len(self._possible_alphas)).to(self._device)
        self._actor_optimizer = get_optimizer(type = self._conf.model.actor_optimizer,
                                              model = self._actor_model,
                                              lr = self._conf.model.actor_init_lr,
                                              weight_decay = self._conf.model.actor_weight_decay                                                                       
                                            )  
        self._actor_loss = ActorLoss()

        self._critic_model = criticNet(state_feature_dim, 1).to(self._device)
        self._critic_optimizer = get_optimizer(type = self._conf.model.critic_optimizer,
                                              model = self._critic_model,
                                              lr = self._conf.model.critic_init_lr,
                                              weight_decay = self._conf.model.critic_weight_decay                                                                       
                                            )          
        self._critic_loss = CriticLoss()

        self._lr_scheduler = get_lr_scheduler(type = self._conf.model.lr_scheduler,
                                              epochs = self._conf.train.episodes,
                                              optimizer = self._actor_optimizer)  
        
        self._episode = 0
        # TODO extend with loading pretrained Actor, Critic


    def train(self):

        while self._episode < self._conf.train.episodes:
            
            # === 2. Initialize Environment ===

            action_batch, state_batch, sparsb_prev, dmapb_prev = self._init_environment()
            # self._model_pruner.reset_model_and_state()

            actions = []
            states = []
            rewards = []
            values = []
            policies = []
            log_probs = []
            entropies = []
            
            # === 3. Iterate Over Layers ===
            for layer_i, layer in enumerate(self._yolo_handler.prunable_layers):

                # --- 3b. Get / Update State ---
                #self._model_pruner.increment_layer()
                #self._model_pruner.update_state()
                state_batch = self._update_state_batch(layer_i, state_batch, sparsb_prev, dmapb_prev)

                # --- 3c. Actorm, Critic Forward ---  
                state_batch_flattened = state_batch.view([self._conf.train.batch_size, -1])
                probs, action_dist, log_softmax = self._actor_model(state_batch_flattened)
                q_value = self._critic_model(state_batch_flattened)

                # --- 3d. Sample Action & Compute Info ---
                action = action_dist.sample()  # alpha index                

                # entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(action).unsqueeze(1)
                policy = probs.gather(-1, action.unsqueeze(0))
                entropy = - (probs * log_softmax).sum(1, keepdim=True)

                for i in range(self._conf.train.batch_size): # TODO: make it more "pythonic"
                    action_batch[i, :, layer_i] = self._possible_alphas[action[layer_i]] # TODO: normalize alpha
                
                # --- 3e. Log Layer Info ---
                # --- 3f. Save Actions ---

                # --- 3g. Predict Error & Sparsity --                
                spn_input_data = torch.cat((action_batch, state_batch), dim=1).view([self._conf.train.batch_size, -1]) # .type(torch.float32).to(device)
                prediction = self._spn_handler.predict(spn_input_data)
                decoded_prediction = self._coder.decode_label(prediction)
                sparsb, dmapb = decoded_prediction['spars'], decoded_prediction['dmap']

                # --- 3h. Compute Reward ---
                reward = self._get_reward(self._conf.reward.type, sparsb, dmapb)

                # --- 3i. Save Trajectory Step ---
                sparsb_prev = sparsb.clone()
                dmapb_prev = dmapb_prev.clone()

                log_probs.append(log_prob)  
                entropies.append(entropy)  
                actions.append(action_batch.clone().detach())
                states.append(state_batch.clone().detach())                
                rewards.append(reward) 
                values.append(q_value)  
                policies.append(policy)  
               

            # === 5. Select Best Result from the batch ==
                
            best_idx = self._coder.decode_label(dmapb[-1, :, 0]).argmin() # TODO list2floatTensor [n_prunableLayers, batch_size, 1]
            best_spars = self._coder.decode_label(states[-1][best_idx, -1, -1]).item()
            best_dmap = self._coder.decode_label(dmapb[-1, best_idx, 0]).item() # TODO list2FloatTensor
            best_alpha_seq = ... # TODO decode alpha denormalize(actions[-1][bidx, 0, :], 0, 2.2)

            # === 6. Compute Returns ===
            returns = self._get_discounted_reward(reward, values, gamma=0.99)

            # === 7. Prepare Log Probs ===
            if self._episode == 0:
                log_probs_prev = torch.zeros(log_probs.shape)
            else:            
                log_probs_prev = log_probs_prev.detach()

            # === 8. Compute Losses ===
            critic_loss = self._critic_loss(rewards, values, 0.99)
            actor_loss = self._actor_loss(rewards, values, policies, log_probs, entropies, ent_coef = self._conf.model.actor_entory_coef, gamma=0.99)
    
            # === 9. Backpropagation ===
            self._actor_optimizer.zero_grad()
            self._critic_optimizer.zero_grad()

            final_loss = actor_loss + critic_loss
            final_loss.backward(retain_graph=True)

            reward_backprop = rewards.mean()
            (-reward_backprop).backward()

            self._actor_optimizer.step()
            self._critic_optimizer.step()

            # === 10. Logging ===

            # === 11. Save Checkpoint ===

            # === 12. End of Episode ===
            self._episode += 1




    def _get_reward(self, reward_type, decoded_sparsb, decoded_dmapb):

        if reward_type == 'proposed':
            reward = reward_function_proposed(decoded_sparsb,
                                              self._conf.reward.target_spars,
                                              decoded_dmapb, 
                                              self._conf.reward.target_dmap,
                                              self._conf.reward.spars_coeff,
                                              self._conf.reward.dmap_coeff,
                                              self._conf.reward.beta )
        elif reward_type == 'purl':
            reward = reward_function_purl(decoded_sparsb,
                                            self._conf.reward.target_spars,
                                            decoded_dmapb, 
                                            self._conf.reward.target_map,
                                            self._conf.reward.map_before, # ? TODO 
                                            self._conf.reward.beta )             

        elif reward_type == 'amc':
            reward = reward_function_amc(decoded_sparsb,
                                         decoded_dmapb)
                                         # init_params TODO

        return reward.unsqueeze(1)
    

    def _get_discounted_reward(self, rewards, gamma):
        """
        Compute normalized discounted cumulative rewards.

        Args:
            rewards (List[Tensor] or Tensor): Sequence of rewards.
            gamma (float): Discount factor.

        Returns:
            Tensor: Normalized discounted reward tensor.
        """
        disc_rewards = []
        val = 0.0
        for i in reversed(range(len(rewards))):
            val = rewards[i] + gamma * val
            disc_rewards.insert(0, val)

        disc_rewards = torch.tensor(disc_rewards, dtype=torch.float32, device=rewards[0].device)
        out = disc_rewards - disc_rewards.mean()
        out /= disc_rewards.std() + 1e-8  # prevent division by zero

        return out

    def _get_advantage(self, rewards, values, gamma=0.99):
        """
        Compute advantage as the difference between discounted rewards and value predictions.

        Args:
            rewards (List[Tensor] or Tensor): Rewards per step.
            values (Tensor): Value function estimates.
            gamma (float): Discount factor.

        Returns:
            Tensor: Advantage values.
        """
        disc_rewards = self._get_discounted_reward(rewards, gamma)
        return disc_rewards - values


    

    def _init_environment(self):

        action_batch = torch.full([self._conf.train.batch_size, 1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        state_batch = torch.full([self._conf.train.batch_size, len(self._state_features)-1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        sparsb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)
        dmapb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)

        return action_batch, state_batch, sparsb_prev, dmapb_prev


    def _update_state_batch(self, layer_i, state_batch, sparsb_prev, dmapb_prev):

        # Only spars and dmap prev.

        state_batch[:, 0, layer_i] = sparsb_prev
        state_batch[:, 1, layer_i] = dmapb_prev

        return state_batch
    

    def _init_action_sequence(self):
        pass

    def _update_action_sequence(self):
        pass



