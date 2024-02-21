import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


def import_script_as_module(config_path) -> Any:
    name = Path(config_path).stem
    spec = spec_from_file_location(name, config_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_dict(d: dict) -> str:
    return json.dumps(d, indent=2, default=lambda o: str(o))
