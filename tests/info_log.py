
import os 
import torch 

base_path = "logs"
list_path = os.listdir(base_path)
for path in list_path:
    x = torch.load(os.path.join(base_path, path)) 
    print("name :", path, "Episodes :", len(x)*20)
    