import os
import torch
import pandas as pd
from torch import Tensor
from typing import List, Tuple
from types import SimpleNamespace
import numpy as np

from utils.tensorboard_handler import TensorboardHandler
from utils.common_utils import normalize, denormalize, set_seed
from src.model.yolo_handler import YoloHandler
from src.model.spn_handler import SPNHandler
from utils.config_parser import ConfigParser
from src.training_components import get_optimizer, get_lr_scheduler, general_cosine_scheduler
#from pruning.model_pruner.stepwise_rl_pruner import StepWiseRLPruner
from pruning.model_pruner.stepwise_pruner import StepWisePruner
from pruning.channel_selection.channel_selector import ChannelSelector
from state_predictor.coder import Coder
from reinforcement_learning.model import actorNet, criticNet
from reinforcement_learning.losses import ActorLoss, CriticLoss, ActorPPOLoss
from reinforcement_learning.rewards import reward_function_proposed, reward_function_purl, reward_function_amc, PartialTargetReward, SigmoidGateReward


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
        set_seed(self._conf.train.seed)

        self._state_features = self._spn_conf.model.state_features

        self._results_df = pd.DataFrame([], columns = ["reward", "spars", "dmap", "alpha_seq"])
        self._best_results_df = pd.DataFrame([], columns = ["reward", "spars", "dmap", "alpha_seq"])

    
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
        spn_handler = SPNHandler(self._spn_conf, run_name=run_name, tb_handler=None)
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

    
    def _save_checkpoint(self):
        checkpoint = {
            'episode': self._episode,
            'actor_state_dict': self._actor_model.state_dict(),
            'critic_stat_dict': self._critic_model.state_dict(),
            'actor_optimizer': self._actor_optimizer.state_dict(),
            'critic_optimizer': self._critic_optimizer.state_dict(),
            'actor_loss': self._actor_loss.state_dict(),
            'critic_loss': self._critic_loss.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(),
            'entropy_values': self._entropy_values
        }
        torch.save(checkpoint, os.path.join(self._log_dir_path, f"checkpoint_{self._episode}.pt"))
        if int(self._best_results_df["reward"].idxmax()) == self._episode:
            torch.save(checkpoint, os.path.join(self._log_dir_path, "checkpoint_best.pt"))
    

    def _load_checkpoint(self, path):
        checkpoint_path = path #os.path.join(path, "checkpoint_best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        self._actor_model.load_state_dict(checkpoint['actor_state_dict'])
        self._critic_model.load_state_dict(checkpoint['critic_stat_dict'])

        self._episode = checkpoint.get('episode', 0)

        if self._conf.model.do_resume:

            self._actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self._critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self._actor_loss.load_state_dict(checkpoint['actor_loss'])
            self._critic_loss.load_state_dict(checkpoint['critic_loss'])

            self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self._entropy_values = checkpoint.get('entropy_values', None)
    
    
    def create(self):

        state_feature_dim = self._n_prunable_layers * (len(self._state_features) - 1) # alpha is not needed -> -1

        self._actor_model = actorNet(state_feature_dim, len(self._possible_alphas)).to(self._device)
        self._actor_optimizer = get_optimizer(type = self._conf.model.actor_optimizer,
                                              model = self._actor_model,
                                              lr = self._conf.model.actor_init_lr,
                                              weight_decay = self._conf.model.actor_weight_decay                                                                       
                                            )  
        
        self._actor_loss = ActorPPOLoss() if self._conf.model.is_ppo else ActorLoss()

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
        
        self._entropy_values = general_cosine_scheduler(min_val = self._conf.model.actor_entropy_coef,
                                                max_val = self._conf.model.entropy_factor * self._conf.model.actor_entropy_coef,
                                                epochs = self._conf.train.episodes,
                                                direction="up")
        
        # Init reward class if needed
        if self._conf.reward.type == 'partial_target':
            self.reward_class = PartialTargetReward( 
                                            self._yolo_handler.prunable_layers,
                                            self._conf.reward.target_spars,
                                            self._conf.reward.target_dmap,
                                            self._conf.reward.beta,
                                            self._conf.reward.spars_coeff,
                                            self._conf.reward.dmap_coeff,
                                            self._device)
            
        elif self._conf.reward.type == 'sigmoid_gate':
            self.reward_class = SigmoidGateReward( 
                                self._yolo_handler.prunable_layers,
                                self._conf.reward.target_spars,
                                self._conf.reward.target_dmap,
                                self._conf.reward.beta,
                                self._conf.reward.spars_coeff,
                                self._conf.reward.dmap_coeff,
                                self._device)

        self._episode = 0

        if len(self._conf.model.pretrained):
            log_path = self._conf.model.pretrained 
            self._load_checkpoint(log_path)

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
            policy_masks = []
            
            # === 3. Iterate Over Layers ===
            for layer_i, layer in enumerate(self._yolo_handler.prunable_layers):
                
                # --- 3c. Actorm, Critic Forward ---  
                state_batch_flattened = state_batch.view([self._conf.train.batch_size, -1])
                probs, action_dist, log_softmax = self._actor_model(state_batch_flattened)
                q_value = self._critic_model(state_batch_flattened)

                # --- 3d. Sample Action & Compute Info ---
                skipmod = getattr(self._conf.train, "skipmod", -1)
                skip_flag = (skipmod > 0) and (layer_i % skipmod != 0)
                    
                sampled_action = action_dist.sample()  # alpha index
                # entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(sampled_action).unsqueeze(1)
                policy = probs.gather(-1, sampled_action.unsqueeze(0))
                entropy = - (probs * log_softmax).sum(1, keepdim=True)

                policy_mask = torch.ones(self._conf.train.batch_size, 1, device=self._device)
                if skip_flag:
                    action = torch.zeros(self._conf.train.batch_size, dtype=int).to(self._device)
                else:
                    action = sampled_action
                    policy_mask.zero_()

                with torch.no_grad():
                    for i in range(self._conf.train.batch_size):
                        alpha_range = [self._possible_alphas[0], self._possible_alphas[-1]]
                        action_batch[i, :, layer_i] = normalize(self._possible_alphas[action[i]], value_range=alpha_range)
                
                # --- 3e. Log Layer Info ---
                self._tb_logging_probs(layer_i, probs)

                # --- 3g. Predict Error & Sparsity --     
                #    spn_input_data shape = [batch_size, n_features * n_prunable_layers]          
                spn_input_data = torch.cat((action_batch, state_batch), dim=1).view([self._conf.train.batch_size, -1]) # .type(torch.float32).to(device)
                spn_input_data = spn_input_data.reshape([spn_input_data.shape[0], len(self._state_features), -1]).T.permute(2,0,1)

                prediction = self._spn_handler.predict(spn_input_data[:, :layer_i+1, :])
                sparsb, dmapb = prediction[0], prediction[1]

                # Update state batc
                if layer_i < self._yolo_handler.n_prunable_layers-1:
                    state_batch = self._update_state_batch(layer_i, state_batch, sparsb, dmapb) 

                # Decode predictions
                decoded_prediction = self._coder.decode_label((sparsb, dmapb)) # Tuple([batch_size], [batch_size])
                decoded_sparsb, decoded_dmapb = decoded_prediction['spars'], decoded_prediction['dmap']

                # --- 3h. Compute Reward ---
                reward = self._get_reward(self._conf.reward.type, layer_i, decoded_sparsb, decoded_dmapb) # [batch_size, 1]

                # --- 3i. Save Trajectory Step ---

                log_probs.append(log_prob)  
                entropies.append(entropy)  
                actions.append(action_batch.clone().detach())
                states.append(state_batch.clone().detach())                
                rewards.append(reward) 
                values.append(q_value)  
                policies.append(policy) 
                policy_masks.append(policy_mask) 
               
            # === 6. Compute Returns ===
            #returns = self._get_discounted_reward(reward, values, gamma=0.99)

            # === 7. Prepare Log Probs ===
            if self._episode == 0 or self._conf.model.pretrained:  
                log_probs_prev = torch.zeros(torch.stack(log_probs).shape)


            # === 8. Compute Losses ===
            critic_loss = self._critic_loss(rewards, values, 0.99)
            if self._conf.model.is_ppo:
                actor_loss = self._actor_loss(rewards, values, policies, torch.stack(log_probs), log_probs_prev.to(self._device), entropies, policy_masks, ent_coef = self._entropy_values[self._episode], gamma=0.99)
            else:
                actor_loss = self._actor_loss(rewards, values, policies, log_probs, entropies, policy_masks, ent_coef = self._entropy_values[self._episode], gamma=0.99)

            log_probs_prev = torch.stack(log_probs).detach()

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
            self._lr_scheduler.step()

            # === 10. Logging ===       
            self._update_results(rewards, states, actions)     
            self._tb_logging(actions[-1], rewards, actor_loss, critic_loss, self._lr_scheduler.get_last_lr()[0], self._entropy_values[self._episode])
            self._folder_logging()

            # === 11. Save Checkpoint ===
            self._save_checkpoint()

            # === 12. End of Episode ===
            self._episode += 1




    def _get_reward(self, reward_type, layer_i, decoded_sparsb, decoded_dmapb):

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
            
        elif reward_type == 'partial_target' or reward_type == 'sigmoid_gate':
            reward = self.reward_class.get_reward(layer_i, decoded_sparsb, decoded_dmapb)

        return reward.unsqueeze(1)
        

    def _init_environment(self):

        action_batch = torch.full([self._conf.train.batch_size, 1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        state_batch = torch.full([self._conf.train.batch_size, len(self._state_features)-1, self._yolo_handler.n_prunable_layers], -1.0).to(self._device)
        sparsb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)
        dmapb_prev = torch.full([self._conf.train.batch_size], -1.0).to(self._device)

        return action_batch, state_batch, sparsb_prev, dmapb_prev
    
    def _double_check_dmap(self, dmap):
        """
        Ensure dmap is not significantly lower than the worst seen so far.
        If any value in dmap is less than 90% of the worst_dmap, it is replaced with the worst_dmap value.
        """
        if self._episode == 0 or self._conf.model.pretrained:
            #self._worst_dmap = torch.full(dmap.size(), -1, dtype=dmap.dtype, device=dmap.device)
            self._worst_dmap = dmap.clone()

        mask = dmap > self._worst_dmap
        self._worst_dmap[mask] = dmap[mask]

        return_dmap = dmap.clone()
        mask = ((self._worst_dmap > 0) & (dmap < 0.9 * self._worst_dmap)) | \
            ((self._worst_dmap < 0) & (dmap > 0.9 * self._worst_dmap))
        return_dmap[mask] = self._worst_dmap[mask]

        return return_dmap


    def _update_state_batch(self, layer_i, state_batch, sparsb_prev, dmapb_prev):

        # Only spars and dmap prev.

        with torch.no_grad():
            state_batch = state_batch.clone()
            state_batch[:, 0, layer_i+1] = sparsb_prev
            state_batch[:, 1, layer_i+1] = dmapb_prev

        return state_batch
    

    def _update_results(self, rewards, states, actions):
        """
        self._results:  [batch_size, 4] -> columns: reward: [1], spars: [1], dmap: [1], alpha_seq: [n_prunable_layers]
                        Overwritten in each episode.

        self._best_results: [n_episodes, 4] -> columns: reward: [1], spars: [1], dmap: [1], alpha_seq: [n_prunable_layers]
                            Expanded with one row in each episode.
        """

        best_idx = states[-1][:, 1, -1].argmin() # best dmap index
        best_results = self._coder.decode_label(states[-1][best_idx, :, -1]) # Tuple (spard, dmap)
        best_alpha_seq = actions[-1][best_idx, 0, :]
        self._best_results_df.loc[self._episode] = [float(sum(r[best_idx, 0] for r in rewards).cpu()), 
                                                    float(best_results['spars']), 
                                                    float(best_results['dmap']), 
                                                    best_alpha_seq.tolist()]

        del self._results_df
        self._results_df = pd.DataFrame({
                "reward": torch.stack([r[:, 0] for r in rewards]).sum(0).cpu().tolist(),
                "spars":  states[-1][:,0,-1].tolist(),
                "dmap": states[-1][:,1,-1].tolist(),
                "alpha_seq": actions[-1][:,0,:].tolist()
            })

    def _tb_logging(self, actions_batch: Tensor, rewards: List[Tensor], actor_loss: Tensor, critic_loss: Tensor, lr: float = None, entropy: float = None):
        
        # Log batch mean and std of action for each prunable layer
        actions_avg = denormalize(torch.mean(actions_batch[:,0,:], dim = 0), value_range=self._samples_conf.alpha.min_max_steps[:2]) # TODO: could be done with self._results_df
        actions_std = denormalize(torch.std(actions_batch[:,0,:], dim = 0), value_range=self._samples_conf.alpha.min_max_steps[:2])
        for i, (avg, std) in enumerate(zip(actions_avg, actions_std)):
            self._tb_handler.log_scalar(avg.item(), self._episode, name=F"mean/layer_{i}", tag_ext="actions")
            self._tb_handler.log_scalar(std.item(), self._episode, name=F"std/layer_{i}", tag_ext="actions")
        
        # Log best results: 
        self._tb_handler.log_scalar(actor_loss.item(), self._episode, name="actor_loss", tag_ext="_results")
        self._tb_handler.log_scalar(critic_loss.item(), self._episode, name="critic_loss", tag_ext="_results")
        self._tb_handler.log_scalar(self._best_results_df.loc[self._episode, 'reward'], self._episode, name=F"_final_reward", tag_ext="_results")
        self._tb_handler.log_scalar(self._best_results_df.loc[self._episode, 'spars'], self._episode, name="best_spars", tag_ext="_results")
        self._tb_handler.log_scalar(self._best_results_df.loc[self._episode, 'dmap'], self._episode, name="best_dmap", tag_ext="_results")

        #Log reward
        for i, reward in enumerate(rewards):
            reward_avg = torch.mean(reward)
            self._tb_handler.log_scalar(reward_avg.item(), self._episode, name=F"layer_{i}", tag_ext="_rewards")

        # Log LR and entropy
        if lr is not None:  self._tb_handler.log_scalar(lr, self._episode, name="learning_rate", tag_ext="_hyperparams")
        if lr is not None:  self._tb_handler.log_scalar(entropy, self._episode, name="entropy", tag_ext="_hyperparams")    



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

    def get_best_results(self):
        """ Returns the episode in which best results were achieved during training, together with the results.
            Used for hyperparameter optimization.
        
        Output: 
            episode (int), reward (float) and corresponding spars (float), dmap (float), alpha_seq (list). 
        """
        best_idx = self._best_results_df['reward'].idxmax()
        best_row = self._best_results_df.loc[best_idx]
        return int(best_idx), float(best_row["reward"]), float(best_row["spars"]), float(best_row["dmap"]), best_row["alpha_seq"]



