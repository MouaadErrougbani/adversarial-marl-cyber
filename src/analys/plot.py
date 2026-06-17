import os
import torch


def load_logs(log_path: str) -> dict:
    """
    Charge tous les fichiers .pt dans log_path.

    Retour:
    {
        "mappo_gat_gat": x,
        "ppo_gat_gat": x,
        ...
    }
    """

    logs = {}

    for filename in os.listdir(log_path):
        if not filename.endswith(".pt"):
            continue

        model_name = filename.replace(".pt", "")
        file_path = os.path.join(log_path, filename)

        x = torch.load(file_path, map_location="cpu")
        logs[model_name] = x

    return logs

logs = load_logs("logs")

for model_name, data in logs.items():
    print(f"Model: {model_name}, Data shape: {len(data)*20}")

