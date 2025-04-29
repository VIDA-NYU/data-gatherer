import json
import importlib.resources as pkg_resources

def load_config(file_name: str):
    with pkg_resources.files('data_gatherer.config').joinpath(file_name).open('r') as f:
        return json.load(f)