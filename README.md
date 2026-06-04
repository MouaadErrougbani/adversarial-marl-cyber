 # Adversarial MARL Cyber

Graph-based multi-agent reinforcement learning (MARL) for CybORG/CAGE4. The project trains five blue agents using PPO and a graph representation of the enterprise network, then packages the trained agents for evaluation and submission.

## Key ideas

- **Graph observations:** The environment is converted into a dynamic graph of hosts, connections, files, and internet nodes.
- **GNN policy:** A GCN-based actor/critic predicts node, edge, and global actions.
- **Multi-agent PPO:** Each blue agent has its own buffer, trained in parallel over batched episodes.
- **CybORG integration:** A wrapper translates model actions into concrete CybORG actions.

## Repository layout

- `train.py`: Parallel rollout generation and PPO training loop.
- `evaluation.py`: Parallel evaluation runner with summary statistics and optional logging.
- `submission.py`: Loads trained weights and exposes the submission agent interface.
- `kaggle_run.py`: Kaggle-oriented setup script (clone, deps, train).
- `models/`: GNN policy, PPO buffer, and utilities.
- `wrapper/`: Graph construction and action translation.

## How it works

1. **Wrap CybORG:** The `GraphWrapper` converts raw observations into a graph state.
2. **Graph encoding:** The GNN encodes nodes and edges plus a global state vector.
3. **Action selection:** The actor outputs node/edge/global actions per agent.
4. **Training:** Rollouts are collected in parallel, then PPO updates run in parallel.
5. **Submission:** Trained weights are loaded and evaluated using the same wrapper.

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train

```bash
python train.py model
```

This writes logs to `logs/` and checkpoints to `checkpoints/`.

### Evaluate

```bash
python evaluation.py
```

The evaluation script loads a submission and runs multiple episodes in parallel.

## Model details

- **Policy:** GCN-based actor/critic with global attention features.
- **Actions:** Node actions, edge actions, and one global action.
- **Memory:** Per-agent PPO buffers aggregated for batched learning.

## Notes

- The environment uses 5 blue agents and fixed episode length defaults.
- Parallelism is controlled via `joblib` and PyTorch thread counts.
- For Kaggle, `kaggle_run.py` handles cloning and dependency setup.
