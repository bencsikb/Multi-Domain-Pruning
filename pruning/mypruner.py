import torch.nn as nn
import torch_pruning as tp
import torch
import gc

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


    def flatten_conv_layers(self, model: nn.Module) -> list:
        flattened_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        return flattened_layers


    def select_indices(self, model: nn.Module) -> dict:
       
        flattened_layers = self.flatten_layers(model)
        indices = {}

        def score_function(layer, alpha: float) -> list:
            
            indices = ... # output channel indices for the given layer
            return indices

        for i, layer in enumerate(flattened_layers):
            indices[f"{i}"] = score_function(layer, self.alpha_sequence[i])
             
        return indices


if __name__ == "__main__":
     
    model = ...
    flattened_conv_layers = ...
    ignored_layers = ...
    alpha_sequence = ... #full 0

    for i, layer in enumerate(flattened_conv_layers):
        if i not in ignored_layers:
            alpha = random()
            alpha_sequence[i] = alpha   

            channel_selector = ChannelSelector(model, alpha_sequence)
            prunabe_output_indices = channel_selector()
            pruner = ModelPruner(model, prunabe_output_indices)
            pruned_model = pruner.prune_model
            del model
            model = pruned_model
    
