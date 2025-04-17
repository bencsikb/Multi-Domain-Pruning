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
from typing import List

from ultralytics.nn.modules import Detect
from src.model.tp_utils import replace_c2f_with_c2f_v2


class ModelHandler:
    def __init__(self, model_conf) -> None:

        # self._flattened_layers = []
        self._model_conf = model_conf
        self._device = model_conf.device
        self._example_input = torch.randn(1, 3, 224, 224) # TODO where should this come from?

        self._init_model = self._load_pretrained()
        self._model = None
        self._detmodel = None 

        self.reset_model()
                
    
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
            
            assert self._model_conf.model_path is not None, "Model path cannot be None"
            model = YOLO(self._model_conf.model_path) 

        elif self._model_conf.pretrained_type == "yolov5":
                model = YOLO('yolov5n.pt') 

        else:
            raise ValueError(f"Model type '{type}' is not supported.")

        return model.to(self._device)

    def reset_model(self) -> None:

        if self._model is not None :
            del self._model

        self._model = copy.deepcopy(self._init_model)
        self._detmodel = self._model.model.train()
        replace_c2f_with_c2f_v2(self._detmodel)
        for _, param in self._detmodel.named_parameters():
            param.requires_grad = True 
        self._model.model = copy.deepcopy(self._detmodel)

        self.determine_prunable_layers()

    
    def determine_prunable_layers(self) -> None: 

        flattened_layers = [(name, module) for name, module in self._detmodel.named_modules() if isinstance(module, nn.Conv2d)]    
        ignored_modules = []
        unwrapped_parameters = []
        for m in self._detmodel.modules():
            if isinstance(m, (Detect,)):
                ignored_modules.append(m)
            
        ignored_layers = [layer for module in ignored_modules for layer in module.modules() if isinstance(layer, nn.Conv2d)]

        self._prunable_layers = [layer for i, layer in enumerate(flattened_layers) if layer[1] not in ignored_layers]



    def train(self):
        pass

    def evaluate(self) -> list:

        prec_metrics = self._model.val(data=self._model_conf.data, batch=self._model_conf.batch_size, plots=None)
        
        M_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        M_params = float(np.around(M_params, 2))
        # TODO calculate flops

        metrics = [float(np.around(m,4)) for m in list(prec_metrics.results_dict.values())[:4]] 
        metrics.append(M_params)

        return metrics # [precision, recall, map50, map95, M_paramns]
    
    def prune(self, all_indices, layer_i):

        self._detmodel.train()

        device = next(self._model.parameters()).device.type
        DG = tp.DependencyGraph().build_dependency(self._detmodel, self._example_input.to(device))

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
                    pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                    pruning_group.prune()

        indices = all_indices[layer_i]     
        layer = self._prunable_layers[layer_i][1] #0:name, 1:layer

        if indices is not None: # and len(indices):   
            prune_conv_layer(layer, indices)
              
        del layer
        gc.collect()

        for name, param in self._detmodel.named_parameters():
            param.requires_grad = True 
        
        self._model.model = copy.deepcopy(self._detmodel)
        

    def save_metrics():
        pass


    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def prunable_layers(self) -> list:
        return self._prunable_layers
    
    @property
    def n_prunable_layers(self) -> int:
        return len(self._prunable_layers)
    
    @property
    def device(self) -> str:
        return self._device

    @property
    def prunable_in_channels(self) -> List[int]:
        return [layer[1].in_channels for layer in self._prunable_layers]
    
    @property
    def prunable_out_channels(self) -> List[int]:
        return [layer[1].out_channels for layer in self._prunable_layers]
    
    @property
    def prunable_kernel_sizes(self) -> List[int]:
        return [layer[1].kernel_size[0] for layer in self._prunable_layers]

    @property
    def prunable_strides(self) -> List[int]:
        return [layer[1].stride[0] for layer in self._prunable_layers]

    @property
    def prunable_paddings(self) -> List[int]:
        return [layer[1].padding[0] for layer in self._prunable_layers]






    



