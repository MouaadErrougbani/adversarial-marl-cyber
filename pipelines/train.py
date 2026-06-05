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
from utils.device import get_device
from models.memory_buffer import MultiPPOMemory
from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph


def default_config():
    return {
        "paths": {
            "logs": "logs",
            "checkpoints": "checkpoints",
        },
        "runtime": {
            "max_threads": 36,
            "max_training_hours": 11.1,
            "device": "auto",
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
    train_device,
):
    print("Updating-v2")

    train_device_str = str(train_device)
    use_xla = train_device_str.startswith("xla")

    if use_xla:
        try:
            import torch_xla.core.xla_model as xm
        except Exception:
            xm = None
            use_xla = False
    else:
        xm = None

    # نقل النماذج إلى جهاز التدريب: TPU/GPU/CPU
    for agent in agents:
        agent.actor.to(train_device)
        agent.critic.to(train_device)
        agent.device = train_device

    if use_xla and xm is not None:
        xm.mark_step()

    last_losses = []

    # تدريب كل agent
    for i in range(agent_count):
        loss = agents[i].learn()
        last_losses.append(loss)

        # مهم مع TPU/XLA:
        # يجبر XLA على تنفيذ العمليات المتراكمة بعد كل agent
        if use_xla and xm is not None:
            xm.mark_step()

    # بعد التدريب نرجع إلى CPU لأن collect_data يستعمل CPU
    for agent in agents:
        agent.actor.to("cpu")
        agent.critic.to("cpu")
        agent.device = torch.device("cpu")

    if use_xla and xm is not None:
        xm.mark_step()

    return last_losses

# def train_models(
#     agents,
#     agent_count,
#     train_device,
# ):
#     print("Updating")

#     # move models to TPU/GPU
#     for agent in agents:
#         agent.actor.to(train_device)
#         agent.critic.to(train_device)
#         agent.device = train_device


#     def learn(i):
#         return agents[i].learn()
    
    

#     # last_losses = Parallel(
#     #     prefer="threads",
#     #     n_jobs=agent_count
#     # )(
#     #     delayed(learn)(i)
#     #     for i in range(agent_count)
#     # )

#     last_losses = []

#     for i in range(agent_count):
#         last_losses.append(
#             agents[i].learn()
#         )

#     for agent in agents:
#         agent.actor.to("cpu")
#         agent.critic.to("cpu")
#         agent.device = torch.device("cpu")



#     return last_losses

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
    train_device,
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
    for e in range(start_iter, total_updates):
        start_ep = e * hp.N
        end_ep = (e + 1) * hp.N

        print("=" * 20, f"Episode {start_ep} -> {end_ep}", "=" * 20)
        
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
            train_device,
        )

        losses = ",".join([f"{last_losses[i]:0.4f}" for i in range(agent_count)])
        print(f"[{e}] Loss: [{losses}]")

        avg_reward = sum(avg_rewards) / hp.N
        print(f"Avg reward for episode: {avg_reward}")
        log.append((avg_reward, e, sum(last_losses) / agent_count))
        torch.save(log, f"{log_dir}/{hp.fnames}.pt")

        for i in range(agent_count):
            agent = agents[i]
            agent.save(outf=f"{checkpoint_dir}/{hp.fnames}-{i}_checkpoint.pt")

            if e % 10_000 < hp.N and e > hp.N:
                agent.save(outf=f"{checkpoint_dir}/{hp.fnames}-{i}_{e // 1000}k.pt")

        elapsed = time.time() - start_time
        if elapsed > max_training_time:
            break


def run_train(cfg):

    seed = cfg["train"]["seed"]

    agent_count = 5

    max_threads = cfg["runtime"]["max_threads"]

    train_device, device_reason = get_device(
        cfg["runtime"].get("device", "auto")
    )

    # CPU pour la collecte
    device = torch.device("cpu")


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
        print(f"Resume enabled: loaded log {log_path} (start_iter={start_iter})")

    override_start = cfg["train"].get("resume_start_iter")
    if override_start is not None:
        start_iter = int(override_start)
        print(f"Resume override start_iter={start_iter}")

    print(f"Training device: {train_device} ({device_reason})")
    print(f"Collection device: {device}")

    try:
        torch.zeros(1, device=device)
    except Exception as exc:
        print(f"Device smoke test failed, falling back to CPU: {exc}")
        device = torch.device("cpu")

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
            epochs=hp.epochs,

            # IMPORTANT :
            # collecte sur CPU
            device=torch.device("cpu"),
        )
        for _ in range(agent_count)
    ]

    if cfg["train"].get("resume"):
        for i, agent in enumerate(agents):
            ckpt_name = cfg["train"].get("resume_name") or cfg["run"]["name"]
            ckpt_path = f"{checkpoint_dir}/{ckpt_name}-{i}_checkpoint.pt"

            if os.path.exists(ckpt_path):
                agent.load_weights(ckpt_path)
                print(f"Checkpoint loaded: agent {i} <- {ckpt_path}")
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
        start_iter=start_iter,
        train_device=train_device,
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



















