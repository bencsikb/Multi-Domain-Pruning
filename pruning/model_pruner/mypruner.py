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
from types import SimpleNamespace
from pruning.channel_selection.channel_selector import ChannelSelector

def prune_layer(model):
     pass




class StepWisePruner():
    def __init__(self, 
                 model_handler: ModelHandler, 
                 init_metrics: dict, 
                 conf: SimpleNamespace, 
                 channel_selector: ChannelSelector) -> None:
                
        self._init_model_handler = model_handler
        self._model_handler = self._init_model_handler
        self._init_metrics = init_metrics
        self.conf = conf
        self.channel_selector = channel_selector

        self.flattened_conv_layers = self._model_handler.flatten_conv_layers
        self.ignored_layers = self.conf.model.ignored_layers
        self.prunable_layers = ... # TODO remove ignored layers from falttened_conv_layers
        self.n_conv_layers = len(self.flattened_conv_layers)
        self.n_prunable_layers = len(flattened_conv_layers) - len(ignored_layers)

        self.example_inputs = torch.randn(1, 3, 224, 224) #TODO
        self._layer_i = 0

        self.state_features = []
        self.metrics_features = []

        self._all_indices = [None] * self.n_prunable_layers
        self._model_state =  pd.DataFrame(0, index=range(self.n_prunable_layers), columns=self.state_features)  # torch.full([self.n_prunable_layers, conf.n_features], -1.0)
        self._label = pd.DataFrame(0, index=range(self.n_prunable_layers), columns=self.metrics_features)# torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        self._alpha_sequence = pd.DataFrame(0, index=range(self.n_prunable_layers), columns=[alpha]) #np.full(self.n_prunable_layers, -1)           

        #TODO call reset model     


    def reset_model(self) -> None:
        """ Resets the model to its original state before applying pruning and imcrements the layer counter.
        Should be called before pruning each layer.
        """
        del self._model_handler
        self._model_handler = self._init_model_handler
        self._layer_i += 1
        self._layer = self.prunable_layers[self._layer_i] # TODO separat func? 

    
    def reset_state(self) -> None:
        """ Resets the state and labels.
        Should be called before pruning the first layer.
        """
        self._all_indices = [None] * self.n_prunable_layers
        self._model_state = torch.full([self.n_prunable_layers, conf.n_features], -1.0)
        self._label = torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        self._alpha_sequence = np.full(self.n_prunable_layers, -1)    

    def select_indices(self) -> None:
        """ Select the indices to be removed from the output dimension, based on the given alpha.
        """
        idxs = channel_selector.select_indices(self.prunable_layers[self._layer_i], self.alpha_sequence.loc[self._layer_i, 'alpha'])
        self._all_indices[self._layer_i] = idxs
   

    def prune_model(self) -> None:   
        """ Prunes the initial model by calling the model_handler's prune function.
        """

        self._model_handler.prune(self._all_indices)
    

    def eval_pruned_model(self) -> None:
        """ Evaluates the model after pruning and updates the metrics after pruning.
        """
        # TODO metrics should be reinitialized somewhere
        self.metrics = self._model_handler.evaluate()


    def update_state_and_label(self) -> None:
                
        self._model_state.loc[self._layer_i, 'in_ch'] = self._layer.in_channels
        self._model_state.loc[self._layer_i, 'out_ch'] = self._layer.out_channels
        self._model_state.loc[self._layer_i, 'kernel'] = self._layer.kernel_size[0]
        self._model_state.loc[self._layer_i, 'stride'] = self._layer.stride[0]
        self._model_state.loc[self._layer_i, 'pad'] = self._layer.padding[0]
        self._model_state.loc[self._layer_i, 'n_pruned_ch'] = len(self._all_indices[self._layer_i]) 
        # TODO all other stuff

        # Model eval
        metrics = self._model_handler.evaluate()
        self._label.loc[self._layer_i, 'recall'] = metrics[0]
        self._label.loc[self._layer_i, 'precision'] = metrics[1]
        self._label.loc[self._layer_i, 'f1'] = metrics[2]
        self._label.loc[self._layer_i, 'map50'] = metrics[3]
        self._label.loc[self._layer_i, 'map90'] = metrics[4]
    
    def fine_tune():
         pass

    def set_alpha(self, alpha) -> None:
        #if self.last_set_alpha + 1 != i:
        #    assert "TODO"
        #else:
        self._alpha_sequence.loc[self._layer_i] = alpha # TODO normalize
    
    @property
    def alpha_sequence(self) -> np.array:
        return self._alpha_sequence

    @property
    def df_to_save(self) -> np.array:
        return pd.concat([self._alpha_sequence, self._model_state, self._label], axis=1)
    
    @property 
    def label(self) -> np.array:
        return self._label #TODO normalize








class OneShotPruner():
        def __init__(self) -> None:
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
    flattened_prunable_layers = ... # TODO
    n_prunable_layers = len(flattened_conv_layers) - len(ignored_layers)

    channel_selector = ChannelSelector(conf.channel_selection)
    pruner = StepWisePruner(model_handler, metrics, conf, channel_selector)

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

        pruner.reset_state()

        for i, layer in enumerate(flattened_prunable_layers):

            # Load model
            pruner.reset_model()
            
            # Check if the alpha_seq exists already
            alpha, is_existing_sample = choose_alpha(pruner.alpha_sequence, i, alpha_pdf, conf, df_allsamples)    

            pruner.set_alpha(i, alpha)  
            pruner.select_indices()              
            pruner.prune_model()
            pruner.eval_pruned_model()
            pruner.update_state_and_label()

            df_to_save = pruner.df_to_save # alpha, model_state, metric labels
            
            # model_handler.finetune(conf.finetune_epochs)
            
            if not is_existing_sample: # Don't save if pruning is only performed to create further non-existing states
                pass
                # save state in pkl
            else:
                # load the labels and check if the saved lables are the same as metrics_after
                # assert if not
                pass

            del model_handler


        
