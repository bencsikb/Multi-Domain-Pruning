import torch.nn as nn
import torch_pruning as tp
import torch
import gc
import sys
import os


class ModelHandler:
    def __init__(self, model_conf) -> None:
        self._model = None
        self._flattened_layers = []
        self._model_conf = model_conf
        self._example_input = torch.randn(1, 3, 224, 224) # TODO where should this come from?

                
    
    def load_pretrained(self, type: str) -> None:
        from ultralytics import YOLOv10

        if type == "yolov10":
            self._model = YOLOv10.from_pretrained('jameslahm/yolov10x')
        else:
            raise ValueError(f"Model type '{type}' is not supported.")

    
    def flatten_conv_layers(self) -> None:

        self._flattened_layers = [module for module in self._model.modules() if isinstance(module, nn.Conv2d)]
    
    def determine_prunable_layers(self) -> None: 
        
        ignored_layer_idxs = self._model_conf.ignored_layers
        self._prunable_layers = [layer for i, layer in enumerate(self._flattened_layers) if i not in ignored_layer_idxs]


    def train(self):
        pass

    def evaluate(self) -> list:

        metrics = self._model.val(data=self._model_conf.data, batch=self._model_conf.batch_size)

        return metrics
    
    def prune(self, all_indices):

        self._model = self._model.model.train()

        DG = tp.DependencyGraph().build_dependency(self._model, self._example_input)

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
                    pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                    pruning_group.prune()

        for i, layer in enumerate(self._prunable_layers):

            indices = all_indices[i]     

            if indices is not None:   
                prune_conv_layer(layer, indices)
            
            del layer
            gc.collect()

        

    def save_metrics():
        pass


    @property
    def model(self) -> nn.Module:
        return self._model

    @model.setter
    def model(self, model: nn.Module) -> None:
        if model is None:
            raise ValueError("Model cannot be set to None.")
        self._model = model # TODO deepcopy?
    
    @property
    def flattened_layers(self) -> list:
        return self._flattened_layers

    @property
    def prunable_layers(self) -> list:
        return self._prunable_layers
    
    @property
    def n_prunable_layers(self) -> int:
        return len(self._prunable_layers)

    



