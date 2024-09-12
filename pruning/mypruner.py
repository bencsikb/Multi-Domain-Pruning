import torch.nn as nn
import torch_pruning as tp
import torch
import gc
import numpy as np

def prune_layer(model):
     pass












class ModelPruner():
    def __init__(self) -> None:
                
        self.model = ...
        self.all_indices = ... # n x list of prunable channels
        self.example_inputs = torch.randn(1, 3, 224, 224)
        self.ignored_layers = ...
    
    def check_layer_compatibility(self, model: nn.Model) -> None:
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


class ChannelSelector():
    """
    Note: also has to be checked: alpha_sequence and ignored_layersm, flattened layers etc match!
    """
    def __init__(self) -> None:
        self.model = ...
        self.ignored_layers = ...
        self.alpha_sequence = ...


    #def flatten_conv_layers(self, model: nn.Module) -> list:
    #    flattened_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    #    return flattened_layers


    def select_indices(self, layer) -> dict:
       
        flattened_layers = self.flatten_layers(model)
        indices = {}

        def score_function(layer, alpha: float) -> list:
            
            indices = ... # output channel indices for the given layer
            return indices

        for i, layer in enumerate(flattened_layers):
            indices[f"{i}"] = score_function(layer, self.alpha_sequence[i])
             
        return indices


import numpy as np

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

    conf = ...
     
    model = ...
    # Evaluate model model.evaluate
    # Save results
    # Get the number of prunable layers
    flattened_conv_layers = ...
    ignored_layers = ...

    n_prunable_layers = len(flattened_conv_layers) - len(ignored_layers)
    alpha_pdf = ... # generate_pdf(n_prunable_layers, len(possible_alphas))

    # Load the samples df and get the n_samples 
    df_allsamples = ...
    n_samples = len(df_allsamples)

    while n_samples < conf.max_samples:


        state = torch.full([n_prunable_layers, conf.n_features], -1.0)
        label = torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        alpha_sequence = np.full(n_prunable_layers, -1)                


        for i, layer in enumerate(flattened_conv_layers):
            if i not in ignored_layers:

                model = ... #load model
                
                # Check if the alpha_seq exists already
                alpha_sequence, is_existing_sample = choose_alpha(alpha_sequence, i, alpha_pdf, conf, df_allsamples)

                        

                channel_selector = ChannelSelector()
                pruner = ModelPruner(model)

                # Eval model before pruning
                if not is_all_sampled:
                    metrics_before = pruner.eval_model(model)

                prunabe_output_indices = channel_selector.select_indices(layer, alpha)
                pruned_model = pruner.prune_model(layer, prunabe_output_indices)
                metrics_after = pruner.eval_model(pruned_model)
                
                finetuned_model = pruner.finetune_model(pruned_model, epochs)
                
                if not is_all_sampled: # Don't save if pruning is only performed to create further non-existing states
                    pruner.save_state(pruned_model)
                else:
                    # load the labels and check if the saved lables are the same as metrics_after
                    # assert if not


                del model
        
