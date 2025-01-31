import torch.nn as nn
import torch_pruning as tp
import torch
import numpy as np
import gc
import sys
import os
from thop import profile
#from fvcore.nn import FlopCountAnalysis
import torchvision.transforms as T
import copy


class ModelHandler:
    def __init__(self, model_conf) -> None:

        self._flattened_layers = []
        self._model_conf = model_conf
        self._device = model_conf.device
        self._example_input = torch.randn(1, 3, 224, 224) # TODO where should this come from?

        self._init_model = self._load_pretrained()
        self._model = copy.deepcopy(self._init_model)
                
    
    def _load_pretrained(self) -> nn.Module:
        from ultralytics import YOLOv10, YOLO

        if self._model_conf.pretrained_type == "yolov10":
            if self._model_conf.model_path is not None: 
                from ultralytics.nn.tasks import attempt_load_one_weight
                from ultralytics.models.yolov10.model import YOLOv10DetectionModel

                model = YOLOv10('yolov10n.yaml') #TODO why n? check!
                weights, ckpt = attempt_load_one_weight(self._model_conf.model_path)
                cfg = ckpt["model"].yaml    
                detmodel = YOLOv10DetectionModel(cfg)
                detmodel.load(weights)
                model.model = copy.deepcopy(detmodel)
            else:
                model = YOLOv10.from_pretrained('jameslahm/yolov10x')

        elif self._model_conf.pretrained_type == "yolov8":

                model = YOLO('yolov8x.pt') 
        else:
            raise ValueError(f"Model type '{type}' is not supported.")

        return model.to(self._device)

    def reset_model(self) -> None:

        del self._model
        self._model = copy.deepcopy(self._init_model)

    
    def flatten_conv_layers(self) -> None:

        self._flattened_layers = [module for module in self._model.modules() if isinstance(module, nn.Conv2d)]
    
    def determine_prunable_layers(self) -> None: 
        
        ignored_layer_idxs = self._model_conf.ignored_layers
        self._prunable_layers = [layer for i, layer in enumerate(self._flattened_layers) if i not in ignored_layer_idxs]


    def train(self):
        pass

    def evaluate(self) -> list:

        prec_metrics = self._model.val(data=self._model_conf.data, batch=self._model_conf.batch_size, plots=None)
        
        M_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        # TODO calculate flops

        metrics = list(prec_metrics.results_dict.values())[:4] + [M_params]

        return metrics # [precision, recall, map50, map95, M_paramns]
    
    def prune(self, all_indices):

        detmodel = self._model.model.train()

        for name, param in detmodel.model.named_parameters():
            param.requires_grad = True 

        device = next(self._model.parameters()).device.type
        DG = tp.DependencyGraph().build_dependency(detmodel, self._example_input.to(device))

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
                    pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                    pruning_group.prune()


        # Determine prunable layers: check if they are in the self._prunable_layers list
        prunable_layers = [module for module in detmodel.modules()
                            if any(
                                module is prunable_layer or (
                                    isinstance(module, type(prunable_layer)) and
                                    all(torch.equal(a, b) for a, b in zip(module.state_dict().values(), prunable_layer.state_dict().values()))
                                )
                                for prunable_layer in self._prunable_layers
                            )]


        for i, layer in enumerate(prunable_layers):

            indices = all_indices[i]     

            if indices is not None: # and len(indices):   
                prune_conv_layer(layer, indices)
              
            del layer
            gc.collect()
        
        self._model.model = detmodel
        

    def save_metrics():
        pass


    @property
    def model(self) -> nn.Module:
        return self._model

    # @model.setter
    # def model(self, model: nn.Module) -> None:
    #     if model is None:
    #         raise ValueError("Model cannot be set to None.")
    #     self._model = model # TODO deepcopy?
    
    @property
    def flattened_layers(self) -> list:
        return self._flattened_layers

    @property
    def prunable_layers(self) -> list:
        return self._prunable_layers
    
    @property
    def n_prunable_layers(self) -> int:
        return len(self._prunable_layers)
    
    @property
    def device(self) -> str:
        return self._device

    



