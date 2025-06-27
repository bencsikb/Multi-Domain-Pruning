import os
import torch
import pandas as pd
from torch import Tensor
from typing import List, Tuple
from types import SimpleNamespace

from utils.tensorboard_handler import TensorboardHandler
from utils.common_utils import normalize, denormalize
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

        self._do_folder_logging = self._conf.train.do_folder_logging 
        self._save_interval = self._conf.train.save_interval 

        # Load configs
        self._spn_conf = self._get_spn_config()
        self._samples_conf = self._get_samples_config()

        self._spn_handler = self._load_spn()
        self._yolo_handler = self._load_yolo()
        self._model_pruner = self._get_model_pruner()
        self._coder = self._initialize_coder()

        self._possible_alphas = self._get_alphas()
        self._n_prunable_layers = self._get_n_prunable_layers()

        self._state_features = self._spn_conf.model.state_features

        self._results_df = pd.DataFrame([], columns = ["spars", "dmap", "alpha_seq"])
        self._best_results_df = pd.DataFrame([], columns = ["spars", "dmap", "alpha_seq"])

    
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

                with torch.no_grad():
                    for i in range(self._conf.train.batch_size):
                        alpha_range = [self._possible_alphas[0], self._possible_alphas[-1]]
                        action_batch[i, :, layer_i] = normalize(self._possible_alphas[action[layer_i]], value_range=alpha_range)
                
                # --- 3e. Log Layer Info ---
                self._tb_logging_probs(layer_i, probs)

                # --- 3g. Predict Error & Sparsity --     
                #    spn_input_data shape = [batch_size, n_features * n_prunable_layers]          
                spn_input_data = torch.cat((action_batch, state_batch), dim=1).view([self._conf.train.batch_size, -1]) # .type(torch.float32).to(device)
                prediction = self._spn_handler.predict(spn_input_data)
                sparsb, dmapb = prediction[0], prediction[1]

                # Update state batc
                state_batch = self._update_state_batch(layer_i, state_batch, sparsb, dmapb) 

                # Decode predictions
                decoded_prediction = self._coder.decode_label(prediction) # Tuple([batch_size], [batch_size])
                decoded_sparsb, decoded_dmapb = decoded_prediction['spars'], decoded_prediction['dmap']

                # --- 3h. Compute Reward ---
                reward = self._get_reward(self._conf.reward.type, decoded_sparsb, decoded_dmapb) # [batch_size, 1]

                # --- 3i. Save Trajectory Step ---

                log_probs.append(log_prob)  
                entropies.append(entropy)  
                actions.append(action_batch.clone().detach())
                states.append(state_batch.clone().detach())                
                rewards.append(reward) 
                values.append(q_value)  
                policies.append(policy)  
               
            # === 6. Compute Returns ===
            #returns = self._get_discounted_reward(reward, values, gamma=0.99)

            # === 7. Prepare Log Probs ===
            if self._episode == 0:  #TODO why do we need this?
                log_probs_prev = torch.zeros(torch.stack(log_probs).shape)
            else:            
                log_probs_prev = log_probs_prev.detach()

            # === 8. Compute Losses ===
            critic_loss = self._critic_loss(rewards, values, 0.99)
            actor_loss = self._actor_loss(rewards, values, policies, log_probs, entropies, ent_coef = self._conf.model.actor_entropy_coef, gamma=0.99)
    
            # === 9. Backpropagation ===
            self._actor_optimizer.zero_grad()
            self._critic_optimizer.zero_grad()

            final_loss = actor_loss + critic_loss
            torch.autograd.set_detect_anomaly(True)
            final_loss.backward(retain_graph=True)

            #TODO reward_backprop = rewards.mean() 
            #(-reward_backprop).backward()

            self._actor_optimizer.step()
            self._critic_optimizer.step()

            # === 10. Logging ===       
            self._update_results(states, actions)     
            self._tb_logging(actions[-1], rewards, actor_loss, critic_loss)
            self._folder_logging()

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
        

    def _init_environment(self):

        action_batch = torch.full([self._conf.train.batch_size, 1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        state_batch = torch.full([self._conf.train.batch_size, len(self._state_features)-1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        sparsb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)
        dmapb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)

        return action_batch, state_batch, sparsb_prev, dmapb_prev


    def _update_state_batch(self, layer_i, state_batch, sparsb_prev, dmapb_prev):

        # Only spars and dmap prev.

        with torch.no_grad():
            state_batch = state_batch.clone()
            state_batch[:, 0, layer_i] = sparsb_prev
            state_batch[:, 1, layer_i] = dmapb_prev

        return state_batch
    

    def _update_results(self, states, actions):
        """
        self._results:  [batch_size, 3] -> columns: spars: [1], dmap: [1], alpha_seq: [n_prunable_layers]
                        Overwritten in each episode.

        self._best_results: [n_episodes, 3] -> columns: spars: [1], dmap: [1], alpha_seq: [n_prunable_layers]
                            Expanded with one row in each episode.
        """

        best_idx = states[-1][:, 1, -1].argmin() # best dmap index
        best_results = self._coder.decode_label(states[-1][best_idx, :, -1]) # Tuple (spard, dmap)
        best_alpha_seq = actions[-1][best_idx, 0, :]
        self._best_results_df.loc[self._episode] = [float(best_results['spars']), float(best_results['dmap']), best_alpha_seq.tolist()]

        del self._results_df
        self._results_df = pd.DataFrame({
                "spars":  states[-1][:,0,-1].tolist(),
                "dmap": states[-1][:,1,-1].tolist(),
                "alpha_seq": actions[-1][:,0,:].tolist()
            })

    def _tb_logging(self, actions_batch: Tensor, rewards: List[Tensor], actor_loss: Tensor, critic_loss: Tensor):
        
        # Log batch mean and std of action for each prunable layer
        actions_avg = denormalize(torch.mean(actions_batch[:,0,:], dim = 0), value_range=self._samples_conf.alpha.min_max_steps[:2]) # TODO: could be done with self._results_df
        actions_std = denormalize(torch.std(actions_batch[:,0,:], dim = 0), value_range=self._samples_conf.alpha.min_max_steps[:2])
        for i, (avg, std) in enumerate(zip(actions_avg, actions_std)):
            self._tb_handler.log_scalar(avg.item(), self._episode, name=F"mean/layer_{i}", tag_ext="actions")
            self._tb_handler.log_scalar(std.item(), self._episode, name=F"std/layer_{i}", tag_ext="actions")
        
        # Log best results: 
        self._tb_handler.log_scalar(actor_loss.item(), self._episode, name="actor_loss", tag_ext="_results")
        self._tb_handler.log_scalar(critic_loss.item(), self._episode, name="critic_loss", tag_ext="_results")
        self._tb_handler.log_scalar(torch.mean(rewards[-1]).item(), self._episode, name=F"_final_reward", tag_ext="_results")
        self._tb_handler.log_scalar(self._best_results_df.loc[self._episode, 'spars'], self._episode, name="best_spars", tag_ext="_results")
        self._tb_handler.log_scalar(self._best_results_df.loc[self._episode, 'dmap'], self._episode, name="best_dmap", tag_ext="_results")

        #Log reward
        for i, reward in enumerate(rewards):
            reward_avg = torch.mean(reward)
            self._tb_handler.log_scalar(reward_avg.item(), self._episode, name=F"layer_{i}", tag_ext="_rewards")



    def _tb_logging_probs(self, layer_i, probs):
        
        # Calculate mean over the barch -> [n_possible_alpha]
        mean_probs = torch.mean(probs, dim=0)
        for i, prob in enumerate(mean_probs):
            self._tb_handler.log_scalar(prob.item(), self._episode, name=F"alpha_{self._possible_alphas[i]}", tag_ext=F"layer_{layer_i}")


    def _folder_logging(self):

        if not self._do_folder_logging:
            return

        folder_path = os.path.join(self._log_dir_path, "logs")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        
        self._best_results_df.to_pickle(os.path.join(folder_path, "bests.pkl")) # TODO handle if resumed
        if self._episode % self._save_interval == 0:
            self._results_df.to_pickle(os.path.join(folder_path, F"{self._episode}.pkl"))
    

    
    def _init_action_sequence(self):
        pass

    def _update_action_sequence(self):
        pass



