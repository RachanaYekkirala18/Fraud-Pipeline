import yaml
from datetime import datetime, timedelta

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# IEEE-CIS: TransactionDT is seconds since a reference; map to a real datetime for ordering/partitioning
BASE_TS = datetime(2017, 12, 1)  # arbitrary but consistent

def txdt_to_datetime(seconds):
    return BASE_TS + timedelta(seconds=int(seconds))
