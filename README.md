export PYTHONPATH=$PYTHONPATH:~/rl_marl/cage-challenge-4
python train.py model_v1
python evaluation.py

python -c "
import torch
x=torch.load('logs/model_v1.pt')
for r,e,l in x:
    print(f'Episodes={e:6d} Reward={r:10.2f} Loss={l:.6f}')
"


nano ~/.kaggle/kaggle.json
{"username":"mouaaderg","key":"0e1c9981a9aea5332dcb3b3d4914437a"}
{"username":"lahcenchbibi","key":"KGAT_f72e371e9e6651499759f37597b659a9"}

kaggle kernels push -p .

rm ~/.kaggle/kaggle.json


KGAT_d58b2ce521de7b589c3eb76f62131651
export KAGGLE_API_TOKEN=KGAT_d58b2ce521de7b589c3eb76f62131651

mkdir -p ~/.kaggle && echo KGAT_d58b2ce521de7b589c3eb76f62131651 > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
