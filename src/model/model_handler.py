import torch.nn as nn
import torch_pruning as tp
import gc


class ModelHandler:
    def __init__(self) -> None:
        self._model = None
        self._flattened_layers = []
        
    
    def load_pretrained(self, type: str) -> None:
        from ultralytics import YOLOv10

        if type == "yolov10":
            self._model = YOLOv10.from_pretrained('jameslahm/yolov10x')
        else:
            raise ValueError(f"Model type '{type}' is not supported.")

    
    def flatten_conv_layers(self) -> None:

        self._flattened_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]

    def train(self):
        pass

    def evaluate(self) -> list:
        pass
        return []
    
    def prune(self, all_indices):

        DG = tp.DependencyGraph().build_dependency(self._model, self.example_inputs)

        def prune_conv_layer(layer: nn.Conv2d, indices: list) -> None:
                    pruning_group = DG.get_pruning_group(layer, tp.prune_conv_out_channels, idxs=indices)
                    pruning_group.prune()

        for i, layer in enumerate(self._flattened_layers):
            
            if i in self.ignored_layers:
                continue

            indices = all_indices[i]        

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


    



