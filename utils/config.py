import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def merge_dicts(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def set_nested_value(cfg, key_path, value):
    cursor = cfg
    for key in key_path[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[key_path[-1]] = value


def apply_overrides(cfg, overrides):
    for raw in overrides or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}', expected key=value")
        key, value = raw.split("=", 1)
        parsed_value = yaml.safe_load(value)
        set_nested_value(cfg, key.split("."), parsed_value)
    return cfg


def load_config(paths, overrides=None):
    cfg = {}
    for path in paths:
        cfg = merge_dicts(cfg, load_yaml(path))
    return apply_overrides(cfg, overrides or [])
