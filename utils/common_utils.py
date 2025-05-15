import torch 
import random 
import numpy as np
import datetime
import uuid


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional: ensure deterministic behavior (slower, but consistent)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_run_name(prefix=""):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    name = f"{date_str}_{short_hash}_{prefix}" if prefix else f"{date_str}_{short_hash}"
    return name