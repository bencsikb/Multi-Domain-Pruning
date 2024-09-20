import torch.nn as nn
import torch_pruning as tp
import torch
import gc
import numpy as np
import pandas as pd
import os

from src.model.model_handler import ModelHandler
from pruning.channel_selection.channel_selector import ChannelSelector
from utils.config_parser import ConfigParser

def prune_layer(model):
     pass












class ModelPruner():
    def __init__(self) -> None:
                
        self.model = ...
        self.all_indices = ... # n x list of prunable channels
        self.example_inputs = torch.randn(1, 3, 224, 224)
        self.ignored_layers = ...
    
    def check_layer_compatibility(self, model: nn.Module) -> None:
        """
        Another idea: all_indices is a dict and the keys are the indices of the prunable layers (in flattened row)
        Then, ignored layers in not even necessary.
        """
        flattened_layers = flattened_layers(model)
        n_conv_layers = len(flattened_layers)
        n_ignored_layers = len(self.ignored_layers)
        n_prunable_layers = len(self.all_indices)

        if n_conv_layers != n_ignored_layers + n_prunable_layers:
             assert "Layer number mismatch!"
    
    def flatten_conv_layers(self, model: nn.Module) -> list:
        flattened_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        return flattened_layers
    

    def prune_model(self, model: nn.Module) -> None:    
        DG = tp.DependencyGraph().build_dependency(model, self.example_inputs)

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
                    pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=[0,1,2,3])
                    pruning_group.prune()

        flattened_layers = self.flatten_layers(model)

        for i, layer in enumerate(flattened_layers):
            
            if i in self.ignored_layers:
                continue

            idxs = self.all_indices[i]        

            prune_conv_layer(layer, idxs)
            
            del layer
            gc.collect()
    
    def fine_tune():
         pass



def choose_alpha(alpha_sequence, i, alpha_pdf, conf, df_allsamples, n_possible_alphas):
    
    def _apply_skip_rules(conf):
        alpha = 1.0
        skipunder = getattr(conf.skiprule, "skipunder", None)
        skipmod = getattr(conf.skiprule, "skipmod", None)

        if (skipunder is not None) and (i < skipunder):
            alpha = 0.0
        elif (skipmod is not None) and (i % skipmod):
            alpha = 0.0

        return alpha

    is_existing_sample = True
    temp_alpha_sequence = alpha_sequence.copy()  # Avoid modifying original sequence directly
    tried_alphas = []
    is_existing_alpha_seq = True  # Initialize the loop control variable

    while is_existing_alpha_seq:
        alpha = _apply_skip_rules(conf)
        if alpha != 0.0:
            alpha = np.random()  # Random value if not skipped

        if alpha not in tried_alphas:
            tried_alphas.append(alpha)
            temp_alpha_sequence[i] = alpha  # Modify only the current index of the sequence
            is_existing_sample = (df_allsamples == temp_alpha_sequence).all(axis=1).any()  # Check if the sequence already exists
            alpha_sequence = temp_alpha_sequence  # Update the main sequence

        if len(tried_alphas) == n_possible_alphas:  # End loop when all alphas have been tried
            is_existing_alpha_seq = False

    return alpha, is_existing_sample




if __name__ == "__main__":

    conf = ConfigParser.read("config/pruning/pruning_sampling.ini")
    
    # Load model
    model_handler = ModelHandler()
    if conf.model.model_path is not None:
        # TODO load model
        pass 
    else:        
        model_handler.load_pretrained(type = conf.model.pretrained_type) 
        # TODO hasattr handling

    # Evaluate model 
    metrics = model_handler.evaluate()
    # Save results

    # Determine prunable layers
    # TODO load model and check of metrics are same as in the generated config file
    model_handler.flatten_conv_layers()
    flattened_conv_layers = model_handler.flattened_layers
    ignored_layers = conf.model.ignored_layers  
    n_prunable_layers = len(flattened_conv_layers) - len(ignored_layers)

    del model_handler

    # Get alpha PDF
    alpha_pdf = ... # generate_pdf(n_prunable_layers, len(possible_alphas))

    # Load the samples df and get the n_samples 
    samples_path = os.path.join(conf.samples.save_path, "allsamples.pkl")
    if os.path.exist(samples_path):
        df_allsamples = pd.read_pkl(samples_path)
        sample_cnt = len(df_allsamples)
    else: 
        df_allsamples = pd.DataFrame(columns=[]) # TODO
        sample_cnt = 0


    while sample_cnt < conf.max_samples:

        state = torch.full([n_prunable_layers, conf.n_features], -1.0)
        label = torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        alpha_sequence = np.full(n_prunable_layers, -1)                


        for i, layer in enumerate(flattened_conv_layers):

            if i in ignored_layers:
                continue

            # Load model
            model_handler = ModelHandler()
            model_handler.load_pretrained(conf.model.pretrained_type) 

            
            # Check if the alpha_seq exists already
            alpha_sequence, is_existing_sample = choose_alpha(alpha_sequence, i, alpha_pdf, conf, df_allsamples)                        

            channel_selector = ChannelSelector(conf.channel_selection)
            pruner = ModelPruner(model_handler.model)

            # Eval model before pruning
            # if not is_existing_sample:
            #    metrics_before = pruner.eval_model(model_handler.model)

            prunabe_output_indices = channel_selector.select_indices(layer, i, alpha_sequence[i])
            model_handler.prune(pruner, layer, prunabe_output_indices)
            metrics_after = model_handler.evaluate()
            
            # model_handler.finetune(conf.finetune_epochs)
            
            if not is_existing_sample: # Don't save if pruning is only performed to create further non-existing states
                pruner.save_state(model_handler.model, state, label, alpha_sequence)
            else:
                # load the labels and check if the saved lables are the same as metrics_after
                # assert if not
                pass

            del model_handler


        
