import pandas as pd
import numpy as np
import logging

from src.model.model_handler import ModelHandler
from src.sample_handler import SampleHandler
from pruning.channel_selection.channel_selector import ChannelSelector
from types import SimpleNamespace

class StepWisePruner():
    def __init__(self, 
                 model_handler: ModelHandler, 
                 sample_handler: SampleHandler,
                 conf: SimpleNamespace, 
                 channel_selector: ChannelSelector) -> None:
                
        self._model_handler = model_handler
        self._sample_handler = sample_handler
        self._conf = conf
        self._channel_selector = channel_selector

        self.state_features = ['is_pruned', 'in_ch', 'out_ch', 'kernel', 'stride', 'pad', 'n_pruned_ch']
        self.label_features = ['recall', 'precision', 'map50', 'map90', 'n_params', 'n_layer_channels']
        self.metrics_features = self.label_features[:-1]

        self._init_metrics = pd.DataFrame(0.0, index=range(1), columns=self.metrics_features)
        self._metrics = pd.DataFrame(0.0, index=range(1), columns=self.metrics_features)

        self._set_init_metrics()

        self._layer_i = -1
        self.reset_model_and_state()
        #TODO call reset model     

    def _set_init_metrics(self) -> None:
        
        init_metrics = self._model_handler.evaluate()
        logging.info(f"Init_metrics: {init_metrics}")

        self._init_metrics.loc[0, 'recall'] = init_metrics[0]
        self._init_metrics.loc[0, 'precision'] = init_metrics[1]
        self._init_metrics.loc[0, 'map50'] = init_metrics[2]
        self._init_metrics.loc[0, 'map90'] = init_metrics[3]
        self._init_metrics.loc[0, 'n_params'] = init_metrics[4]
    
    def _set_metrics(self, metrics) -> None:
        
        self._metrics.loc[0, 'recall'] = metrics[0]
        self._metrics.loc[0, 'precision'] = metrics[1]
        self._metrics.loc[0, 'map50'] = metrics[2]
        self._metrics.loc[0, 'map90'] = metrics[3]
        self._metrics.loc[0, 'n_params'] = metrics[4]


    def increment_layer(self) -> None:
        """ 
        Increments the layer counter and takes the next prunable layer.
        Should be called before pruning each layer.
        """
        self._layer_i += 1
        self._layer = self._model_handler.prunable_layers[self._layer_i] 

    
    def reset_model_and_state(self) -> None:
        """ 
        Resets the model, state and labels.
        Should be called before pruning the first layer.
        """
        self._model_handler.reset_model()
        self._layer_i = -1
        self._metrics[self._metrics.columns] = self._init_metrics.values
        self._all_indices = [None] * self._model_handler.n_prunable_layers
        self._model_state =  pd.DataFrame(0, index=range(self._model_handler.n_prunable_layers), columns=self.state_features) 
        self.update_state() # fill the channel-related params
        self._label = pd.DataFrame(0.0, index=range(self._model_handler.n_prunable_layers), columns=self.metrics_features + [col + '_init' for col in self.metrics_features])
        self._alpha_sequence = pd.DataFrame(0.0, index=range(self._model_handler.n_prunable_layers), columns=['alpha'])         


    def select_indices(self) -> None:
        """ Select the indices to be removed from the output dimension, based on the given alpha.
        """
        idxs = self._channel_selector.select_indices(self._model_handler.prunable_layers[self._layer_i][1], self.alpha_sequence.loc[self._layer_i, 'alpha'])
        self._all_indices[self._layer_i] = idxs
        logging.info(f"{len(idxs)} / {self._model_handler.prunable_layers[self._layer_i][1].out_channels} channels will be removed.")
   

    def prune_model(self) -> None:   
        """ Prunes the initial model by calling the model_handler's prune function.
        """
        if self._all_indices[self._layer_i] is not None and self._all_indices[self._layer_i]: # prune only if there is sth to prune
            self._model_handler.prune(self._all_indices, self._layer_i)
            self._model_handler.determine_prunable_layers()
    

    def eval_pruned_model(self) -> None:
        """ Evaluates the model after pruning and updates the metrics after pruning.
        """
        # TODO metrics should be reinitialized somewhere
        if self._all_indices[self._layer_i] is not None and self._all_indices[self._layer_i]:
            metrics = self._model_handler.evaluate()
            self._set_metrics(metrics)
            logging.info(f"Metrics: {metrics}")


    def update_state(self) -> None:
               
        self._model_state.loc[:, 'in_ch'] = self._model_handler.prunable_in_channels
        self._model_state.loc[:, 'out_ch'] = self._model_handler.prunable_out_channels
        self._model_state.loc[:, 'kernel'] = self._model_handler.prunable_kernel_sizes
        self._model_state.loc[:, 'stride'] = self._model_handler.prunable_strides
        self._model_state.loc[:, 'pad'] = self._model_handler.prunable_paddings

        if self._layer_i  >= 1:
            self._model_state.loc[self._layer_i-1, 'is_pruned'] = 1
            self._model_state.loc[self._layer_i-1, 'n_pruned_ch'] = len(self._all_indices[self._layer_i-1]) 

    def update_label(self, is_existing_sample) -> None:

        if not is_existing_sample: 

            self._label.loc[self._layer_i, self.metrics_features] = self._metrics.values
            self._label.loc[self._layer_i, 'n_layer_channels'] = self._layer[1].out_channels

            init_columns = [col + '_init' for col in self.metrics_features]
            self._label.loc[self._layer_i, init_columns] = self._init_metrics.values.flatten()
        
        else:
            saved_label_df = self._sample_handler.retrieve_sample(self.data)
            assert self._sample_handler.check_label_equality(saved_label_df, self._init_metrics), (
               "Label equality check failed. The saved label DataFrame does not match the initial metrics.")
            self._label.iloc[self._layer_i] = saved_label_df
    
    
    def fine_tune():
         pass

    def set_alpha(self, alpha) -> None:
        self._alpha_sequence.loc[self._layer_i, 'alpha'] = alpha 
    
    @property
    def alpha_sequence(self) -> pd.DataFrame:
        return self._alpha_sequence

    @property
    def data(self) -> pd.DataFrame:
        data_df = pd.concat([self._alpha_sequence, self._model_state, ], axis=1)
        return data_df
    
    @property 
    def label(self) -> pd.DataFrame:
        return self._label.iloc[[self._layer_i]]
