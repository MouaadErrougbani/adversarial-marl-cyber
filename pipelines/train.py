# pipelines/train.py
from argparse import ArgumentParser
import os
import time
from types import SimpleNamespace
import warnings
from tqdm import tqdm

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings(
    "ignore",
    message=".*Gym has been unmaintained since 2022.*",
)

from joblib import Parallel, delayed
import torch

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from models.cage4 import InductiveGraphPPOAgent
from models.memory_buffer import MultiPPOMemory
from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph

from utils.device import get_device



def default_config():
    return {
        "paths": {
            "logs": "logs",
            "checkpoints": "checkpoints",
        },
        "runtime": {
            "max_threads": 36,
            "max_training_hours": 11.1,
        },
        "train": {
            "seed": 1337,
            "episode_len": 500,
            "episodes_per_update": 25,
            "workers": 25,
            "batch_size": 2500,
            "training_episodes": 50_000,
            "epochs": 4,
            "resume": False,
            "resume_name": None,
            "resume_start_iter": None,
        },
        "model": {
            "hidden": 256,
            "embedding": 128,
            "actor_lr": 0.0003,
            "critic_lr": 0.001,
            "clip": 0.2,
        },
        "run": {
            "name": "model",
        },
    }


def build_hyper_params(cfg):
    return SimpleNamespace(
        N=cfg["train"]["episodes_per_update"],
        workers=cfg["train"]["workers"],
        bs=cfg["train"]["batch_size"],
        episode_len=cfg["train"]["episode_len"],
        training_episodes=cfg["train"]["training_episodes"],
        epochs=cfg["train"]["epochs"],
        fnames=cfg["run"]["name"],
    )


@torch.no_grad()
def generate_episode_job(agents, env, hp, agent_count, max_threads, i):
    torch.set_num_threads(max_threads // hp.workers)

    env.reset()
    states = env.last_obs
    blocked_rewards = [0] * agent_count

    tot_reward = 0
    memory_buffers = MultiPPOMemory(hp.bs, agents=agent_count)

    for ts in tqdm(range(hp.episode_len), desc="Generating episode"):
        actions = dict()
        memories = dict()

        for k, (state, blocked) in states.items():
            i = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action, value, prob = agents[i].get_action((state, blocked))
                memories[i] = (state, action, value, prob)
                actions[k] = action

        next_state, rewards, _, _, _ = env.step(actions)
        rewards = list(rewards.values())
        tot_reward += sum(rewards) / agent_count

        for i in range(agent_count):
            if i in memories:
                s, a, v, p = memories[i]
                r = rewards[i] + blocked_rewards[i]
                t = 0 if ts < hp.episode_len - 1 else 1

                memory_buffers.remember(i, s, a, v, p, r, t)
                blocked_rewards[i] = 0
            else:
                blocked_rewards[i] += rewards[i]

        states = next_state

    return memory_buffers.mems, tot_reward


def collect_data(
    agents,
    envs,
    hp,
    agent_count,
    max_threads,
):

    out = Parallel(prefer="processes", n_jobs=hp.workers)(
            delayed(generate_episode_job)(agents, envs[i % len(envs)], hp, agent_count, max_threads, i)
            for i in range(hp.N)
        )

    return out


def train_models(
    agents,
    agent_count,
):
    print("Updating", flush=True)

    def learn(i):
        return agents[i].learn()
    
    import time
    start_time = time.time()


    last_losses = Parallel(
        prefer="threads",
        n_jobs=agent_count
    )(
        delayed(learn)(i)
        for i in range(agent_count)
    )
    print(f"Update time: {time.time() - start_time:0.2f} seconds", flush=True)

    return last_losses


def test_tpu_xla(size=2048, steps=20, force_pjrt=True):
    """
    Teste si TPU/XLA fonctionne avec PyTorch/XLA.

    Args:
        size: taille de la matrice, ex: 2048 => 2048x2048
        steps: nombre d'itérations du benchmark
        force_pjrt: si True, met PJRT_DEVICE=TPU avant import torch_xla

    Returns:
        dict avec les informations du test
    """

    if force_pjrt:
        # Important: doit être fait avant import torch_xla
        os.environ["PJRT_DEVICE"] = "TPU"

    import torch
    import torch_xla.core.xla_model as xm

    print("🔍 Testing TPU/XLA...", flush=True)

    device = xm.xla_device()
    print("Device:", device, flush=True)

    # Petit tensor test
    x = torch.ones((4, 4), device=device)
    y = x @ x

    xm.mark_step()

    print("Small tensor device:", y.device, flush=True)
    print("Small result:", flush=True)
    print(y.cpu(), flush=True)

    print("\n🚀 Running matrix multiplication benchmark on TPU...", flush=True)

    a = torch.randn((size, size), device=device)
    b = torch.randn((size, size), device=device)

    # Warmup: première exécution compile souvent
    t0 = time.time()
    c = a @ b
    xm.mark_step()
    warmup_time = time.time() - t0

    print(f"Warmup time: {warmup_time:.2f}s", flush=True)

    # Benchmark
    t0 = time.time()

    for _ in range(steps):
        c = a @ b
        xm.mark_step()

    total_time = time.time() - t0
    avg_time = total_time / steps

    print(f"Benchmark steps: {steps}", flush=True)
    print(f"Matrix size: {size}x{size}", flush=True)
    print(f"Total time: {total_time:.2f}s", flush=True)
    print(f"Average time per step: {avg_time:.4f}s", flush=True)

    # Ramener une petite valeur vers CPU pour confirmer résultat réel
    result = c[0, 0].detach().cpu().item()
    print("Result sample:", result, flush=True)

    print("✅ TPU test finished successfully", flush=True)

    return {
        "device": str(device),
        "size": size,
        "steps": steps,
        "warmup_time": warmup_time,
        "total_time": total_time,
        "avg_time_per_step": avg_time,
        "result_sample": result,
    }


def train(
    agents,
    hp,
    seed,
    agent_count,
    max_threads,
    max_training_hours,
    log_dir,
    checkpoint_dir,
    log,
    start_iter,
):
    for agent in agents:
        agent.train()
    envs = []
    for _ in range(min(hp.workers, hp.N)):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=hp.episode_len,
        )
        env = CybORG(sg, "sim", seed=seed)
        envs.append(GraphWrapper(env))


    max_training_time = max_training_hours * 60 * 60
    start_time = time.time()

    total_updates = hp.training_episodes // hp.N
    device, reason = get_device("xla")
    print("Device:", device)
    print("Reason:", reason)
    for e in range(start_iter, total_updates):
        start_ep = e * hp.N
        end_ep = (e + 1) * hp.N

        print("=" * 20, f"Episode {start_ep} -> {end_ep}", "=" * 20, flush=True)
        
        out = collect_data(
            agents,
            envs,
            hp,
            agent_count,
            max_threads,
        )

        memories, avg_rewards = zip(*out)
        memories = [list(m) for m in zip(*memories)]
        for i in range(agent_count):
            agents[i].memory.mems = memories[i]

        last_losses = train_models(
            agents,
            agent_count,
        )

        losses = ",".join([f"{last_losses[i]:0.4f}" for i in range(agent_count)])
        print(f"[{e}] Loss: [{losses}]", flush=True)

        avg_reward = sum(avg_rewards) / hp.N
        print(f"Avg reward for episode: {avg_reward}", flush=True)
        log.append((avg_reward, e, sum(last_losses) / agent_count))
        torch.save(log, f"{log_dir}/{hp.fnames}.pt")

        for i in range(agent_count):
            agent = agents[i]
            agent.save(outf=f"{checkpoint_dir}/{hp.fnames}-{i}_checkpoint.pt")

            if e % 10_000 < hp.N and e > hp.N:
                agent.save(outf=f"{checkpoint_dir}/{hp.fnames}-{i}_{e // 1000}k.pt")

        elapsed = time.time() - start_time
        
        # if str(device).startswith("xla"):
        #     test_tpu_xla(size=2048, steps=20, force_pjrt=False)
        if elapsed > max_training_time:
            break


def run_train(cfg):

    seed = cfg["train"]["seed"]

    agent_count = 5

    max_threads = cfg["runtime"]["max_threads"]


    torch.manual_seed(seed)
    torch.set_num_threads(max_threads)

    hp = build_hyper_params(cfg)

    log_dir = cfg["paths"]["logs"]
    checkpoint_dir = cfg["paths"]["checkpoints"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    resume_name = cfg["train"].get("resume_name") or cfg["run"]["name"]
    log_path = f"{log_dir}/{resume_name}.pt"

    log = []
    start_iter = 0

    if cfg["train"].get("resume") and os.path.exists(log_path):
        log = torch.load(log_path, map_location="cpu")
        start_iter = len(log)
        print(f"Resume enabled: loaded log {log_path} (start_iter={start_iter})", flush=True)

    override_start = cfg["train"].get("resume_start_iter")
    if override_start is not None:
        start_iter = int(override_start)
        print(f"Resume override start_iter={start_iter}", flush=True)


    agents = [
        InductiveGraphPPOAgent(
            ObservationGraph.DIM + 5,
            bs=hp.bs,
            a_kwargs={
                "lr": cfg["model"]["actor_lr"],
                "hidden1": cfg["model"]["hidden"],
                "hidden2": cfg["model"]["embedding"],
            },
            c_kwargs={
                "lr": cfg["model"]["critic_lr"],
                "hidden1": cfg["model"]["hidden"],
                "hidden2": cfg["model"]["embedding"],
            },
            clip=cfg["model"]["clip"],
            epochs=hp.epochs
            
        )
        for _ in range(agent_count)
    ]

    if cfg["train"].get("resume"):
        for i, agent in enumerate(agents):
            ckpt_name = cfg["train"].get("resume_name") or cfg["run"]["name"]
            ckpt_path = f"{checkpoint_dir}/{ckpt_name}-{i}_checkpoint.pt"

            if os.path.exists(ckpt_path):
                agent.load_weights(ckpt_path)
                print(f"Checkpoint loaded: agent {i} <- {ckpt_path}", flush=True)
            else:
                print(f"Warning: checkpoint not found for agent {i}: {ckpt_path}")

    print("==" * 40)
    print("Hyperparameters:")
    print("Seed:", seed)
    print("Episode length:", hp.episode_len)
    print("Episodes per update:", hp.N)
    print("Training episodes:", hp.training_episodes)
    print("Workers:", hp.workers)
    print("Batch size:", hp.bs)
    print("Hidden layer dimension:", cfg["model"]["hidden"])
    print("Embedding dimension:", cfg["model"]["embedding"])
    print("==" * 40)

    train(
        agents,
        hp,
        seed=seed,
        agent_count=agent_count,
        max_threads=max_threads,
        max_training_hours=cfg["runtime"]["max_training_hours"],
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        log=log,
        start_iter=start_iter
    )

def main_legacy():
    ap = ArgumentParser()
    ap.add_argument("fname", help="Required: the name to save output files as.")
    ap.add_argument(
        "--hidden",
        action="store",
        type=int,
        default=256,
        help="Dimension of middle layer for actor/critic",
    )
    ap.add_argument(
        "--embedding",
        action="store",
        type=int,
        default=128,
        help="Dimension of node representation for actor/critic",
    )

    args = ap.parse_args()
    cfg = default_config()
    cfg["run"]["name"] = args.fname
    cfg["model"]["hidden"] = args.hidden
    cfg["model"]["embedding"] = args.embedding

    run_train(cfg)


if __name__ == "__main__":
    main_legacy()



















