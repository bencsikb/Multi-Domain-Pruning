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

        self._layer_i = -1
        self._metrics = None

        self.state_features = []
        self.metrics_features = []

        self._all_indices = [None] * self._model_handler.n_prunable_layers
        self._model_state =  pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=self.state_features)  # torch.full([self.n_prunable_layers, conf.n_features], -1.0)
        self._label = pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=self.metrics_features)# torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        self._alpha_sequence = pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=['alpha']) #np.full(self.n_prunable_layers, -1)           

        #TODO call reset model     


    def reset_model(self) -> None:
        """ Resets the model to its original state before applying pruning and imcrements the layer counter.
        Should be called before pruning each layer.
        """
        del self._model_handler
        self._model_handler = self._init_model_handler
        self._layer_i += 1
        self._layer = self._model_handler.prunable_layers[self._layer_i] # TODO self.prunable_layers[self._layer_i] # TODO separat func? 

    
    def reset_state(self) -> None:
        """ Resets the state and labels.
        Should be called before pruning the first layer.
        """
        self._layer_i = -1
        self._all_indices = [None] * self._model_handler.n_prunable_layers
        self._model_state =  pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=self.state_features)  # torch.full([self.n_prunable_layers, conf.n_features], -1.0)
        self._label = pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=self.metrics_features)# torch.zeros([1, 4])  # sparsity, dmap, drec, dprec
        self._alpha_sequence = pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=['alpha']) #np.full(self.n_prunable_layers, -1)           


    def select_indices(self) -> None:
        """ Select the indices to be removed from the output dimension, based on the given alpha.
        """
        idxs = channel_selector.select_indices(self._model_handler.prunable_layers[self._layer_i], self.alpha_sequence.loc[self._layer_i, 'alpha'])
        #idxs = channel_selector.select_indices(self.flattened_conv_layers[self._layer_i], self._alpha_sequence.loc[self._layer_i, 'alpha'])

        self._all_indices[self._layer_i] = idxs
   

    def prune_model(self) -> None:   
        """ Prunes the initial model by calling the model_handler's prune function.
        """
        if self._all_indices[self._layer_i] is not None and self._all_indices[self._layer_i]:
            self._model_handler.prune(self._all_indices)
    

    def eval_pruned_model(self) -> None:
        """ Evaluates the model after pruning and updates the metrics after pruning.
        """
        # TODO metrics should be reinitialized somewhere
        self._metrics = self._model_handler.evaluate()


    def update_state_and_label(self) -> None:
                
        self._model_state.loc[self._layer_i, 'in_ch'] = self._layer.in_channels
        self._model_state.loc[self._layer_i, 'out_ch'] = self._layer.out_channels
        self._model_state.loc[self._layer_i, 'kernel'] = self._layer.kernel_size[0]
        self._model_state.loc[self._layer_i, 'stride'] = self._layer.stride[0]
        self._model_state.loc[self._layer_i, 'pad'] = self._layer.padding[0]
        self._model_state.loc[self._layer_i, 'n_pruned_ch'] = len(self._all_indices[self._layer_i]) 
        # TODO all other stuff

        # Model eval
        if self._metrics is None:
            assert "Metrics after pruning are missing. Function \"eval_pruned_model\" has to be called first!"
        self._label.loc[self._layer_i, 'recall'] = self._metrics[0]
        self._label.loc[self._layer_i, 'precision'] = self._metrics[1]
        #self._label.loc[self._layer_i, 'f1'] = self._metrics[2]
        self._label.loc[self._layer_i, 'map50'] = self._metrics[2]
        self._label.loc[self._layer_i, 'map90'] = self._metrics[3]
    
    def fine_tune():
         pass

    def set_alpha(self, alpha) -> None:
        #if self.last_set_alpha + 1 != i:
        #    assert "TODO"
        #else:
        self._alpha_sequence.loc[self._layer_i, 'alpha'] = alpha # TODO normalize
    
    @property
    def alpha_sequence(self) -> pd.DataFrame:
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


def choose_alpha(alpha_sequence, i, alpha_pdf, conf, sample_handler):
    
    def _apply_skip_rules(conf):
        alpha = 1.0
        skipunder = getattr(conf.channel_selection, "skipunder", None)
        skipmod = getattr(conf.channel_selection, "skipmod", None)

        if (skipunder is not None) and (i < skipunder):
            alpha = 0.0
        elif (skipmod is not None) and (i % skipmod):
            alpha = 0.0

        return alpha

    #TODO This sould actually go to the PDF generator
    if getattr(conf.alpha, 'value_list', None) is not None:
        possible_alphas = conf.alpha.value_list
    else:
        assert hasattr(conf.alpha, 'min_max_step'), "Alpha value list OR min, max, step values must be provided!"
        possible_alphas = np.arange(conf.alpha.min_max_step[0], conf.alpha.min_max_step[1], conf.alpha.min_max_step[2])
    n_possible_alphas = len(possible_alphas)

    alpha_seq_df = alpha_sequence.copy()
    tried_alphas = []
    is_existing_sample = True
    while is_existing_sample:
        alpha = _apply_skip_rules(conf)
        if alpha != 0.0:
            alpha = np.random()  # Random value if not skipped
        
        if alpha not in tried_alphas:
            tried_alphas.append(alpha)
            alpha_seq_df.loc[i, 'alpha'] = alpha  
            is_existing_sample = sample_handler.is_existing_sample(alpha_seq_df)
        
        if len(tried_alphas) == n_possible_alphas:  
                break              

    return alpha, is_existing_sample



class SampleHandler():
    def __init__(self, conf: SimpleNamespace) -> None:
        self.samples_path = conf.samples.save_path

        self.sample_container = set()
    
    def read_all_samples(self) -> None:

        for filename in os.listdir(self.samples_path):
            if filename.endswith('.pkl'):
                sample_df = pd.read_pickle(os.path.join(self.samples_path, filename))
                alpha_seq = sample_df['alpha']
                self.add_sample(alpha_seq)               


    def is_existing_sample(self, alpha_df) -> bool:
        sample_tuple = self.df_to_tuple(alpha_df)
        is_exists = True if sample_tuple in self.sample_container else False
        return is_exists


    def add_sample(self, sample_df) -> None:
        alpha_df = sample_df['alpha']
        sample_tuple = self.df_to_tuple(alpha_df)
        self.sample_container.add(sample_tuple)

    def df_to_tuple(self, df):
        return tuple(map(tuple, df.values))
    
    @property
    def n_samples(self) -> int:
        return len(self.sample_container)


if __name__ == "__main__":

    conf = ConfigParser.read("config/pruning/pruning_sampling.ini")
    
    # Load model
    model_handler = ModelHandler(conf.model)
    if conf.model.model_path is not None:
        # TODO load model
        pass 
    else:        
        model_handler.load_pretrained(type = conf.model.pretrained_type) 
        # TODO hasattr handling

    # Evaluate model 
    # TODO metrics = model_handler.evaluate()
    # Save results

    # Determine prunable layers
    # TODO load model and check of metrics are same as in the generated config file
    metrics = []
    model_handler.flatten_conv_layers()
    model_handler.determine_prunable_layers()
    prunable_layers = model_handler.prunable_layers
  
    channel_selector = ChannelSelector(conf.channel_selection)
    pruner = StepWisePruner(model_handler, metrics, conf, channel_selector)

    del model_handler

    # Get alpha PDF
    alpha_pdf = ... # generate_pdf(n_prunable_layers, len(possible_alphas))

    # Load the samples df and get the n_samples 
    sample_handler = SampleHandler(conf)
    sample_handler.read_all_samples()
    sample_cnt = sample_handler.n_samples

    while sample_cnt < conf.samples.max_samples:

        pruner.reset_state()

        for i, layer in enumerate(prunable_layers):

            # Load model
            pruner.reset_model()
            
            # Check if the alpha_seq exists already
            alpha, is_existing_sample = choose_alpha(pruner.alpha_sequence, i, None, conf, sample_handler)    # TODO remove sample dependency

            pruner.set_alpha(alpha)  
            pruner.select_indices()              
            pruner.prune_model()
            pruner.eval_pruned_model()
            pruner.update_state_and_label()

            df_to_save = pruner.df_to_save # alpha, model_state, metric labels
            
            # model_handler.finetune(conf.finetune_epochs)
            
            if is_existing_sample: # Don't save if pruning is only performed to create further non-existing states
                print("The state already exists in the dataset.") # TODO log
                continue
                # load the labels and check if the saved lables are the same as metrics_after
                # assert if not
            else:
                sample_save_path = os.path.join(conf.samples.save_path, str(sample_cnt) + ".pkl")
                if os.path.exists(sample_save_path):
                    assert f"Sample {sample_cnt} already exists!"
                df_to_save.to_pickle(sample_save_path)
                sample_handler.add_sample(df_to_save)

            del model_handler


        
