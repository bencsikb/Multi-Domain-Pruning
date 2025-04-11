import os

from utils.tensorboard_handler import TensorboardHandler
from src.model.yolo_handler import YoloHandler
from src.model.spn_handler import SPNHandler


class RLAgent():
    def __init__(self, conf, run_name: str, tb_handler: TensorboardHandler) -> None:
        
        self._conf = conf
        self._run_name = run_name
        self._tb_handler = tb_handler
        self._log_dir_path = os.path.join(conf.save.root, run_name)

        self._spn_handler = self._load_spn()
        self._yolo_handler = self._load_yolo()


    def _load_spn(self) -> SPNHandler:
        
        run_path = self._conf.spn.root        
        run_name = os.path.basename(run_path)
        conf_path = os.path.join(run_path, "settings.ini")
        # load or define SPN model
        spn_handler = SPNHandler(conf_path, run_name=run_name)
        spn_handler.create(is_pretrained=True)

        return spn_handler
    
    
    def _load_yolo(self) -> YoloHandler:

        yolo_conf = self._conf.yolo
        yolo_handler = YoloHandler(yolo_conf)

        return yolo_handler        
    
    
    def create(self):
        pass

