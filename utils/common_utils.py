import datetime
import uuid


def generate_run_name(prefix=""):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = uuid.uuid4().hex[:6]
    name = f"{date_str}_{short_hash}_{prefix}" if prefix else f"{date_str}_{short_hash}"
    return name