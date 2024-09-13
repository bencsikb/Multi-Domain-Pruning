import torch.nn as nn


class ModelHandler:
    def __init__(self) -> None:
        self._model = None
        self._flattened_layers = []
        
    
    def load_pretrained(self, type) -> None:
        from ultralytics import YOLOv10

        if type == "yolov10":
            self._model = YOLOv10.from_pretrained('jameslahm/yolov10x')
        else:
            raise ValueError(f"Model type '{type}' is not supported.")

    
    def flatten_conv_layers(self) -> None:

        self._flattened_layers = [module for module in self.model.modules() if isinstance(module, nn.Conv2d)]

    def train(self):
        pass

    def validate(self):
        pass

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


    



