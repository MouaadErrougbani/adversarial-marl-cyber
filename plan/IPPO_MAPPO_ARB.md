# Passage de IPPO vers MAPPO

Ce document explique, étape par étape, comment transformer l’architecture actuelle du projet de **IPPO** vers **MAPPO**.

L’explication est basée sur l’architecture actuelle du projet :

```text
src/
├── environments/
├── observations/
├── models/
├── trainers/
└── utils/
```

---

## 1. الفرق بين IPPO و MAPPO

### IPPO actuel

في الوضع الحالي، المشروع يستعمل **IPPO**:

```text
Agent 0 -> Actor 0 + Critic 0
Agent 1 -> Actor 1 + Critic 1
Agent 2 -> Actor 2 + Critic 2
Agent 3 -> Actor 3 + Critic 3
Agent 4 -> Actor 4 + Critic 4
```

كل Agent يتعلم باستعمال ملاحظته المحلية فقط.

يعني:

```text
Actor_i(local_obs_i)  -> action_i
Critic_i(local_obs_i) -> value_i
```

---

### MAPPO المطلوب

في MAPPO نستعمل:

```text
Actor décentralisé
Critic centralisé
```

يعني:

```text
Actor_i(local_obs_i) -> action_i
Critic(global_state) -> value
```

الـ Actor يبقى محلياً، لكن الـ Critic يرى حالة أوسع أو مشتركة لكل agents.

---

## 2. الملفات التي لا تحتاج تغيير كبير

هذه الملفات غالباً تبقى كما هي:

```text
src/environments/cage4/
├── constants.py
├── topology.py
├── action_space.py
├── action_translator.py
├── env_factory.py
├── graph_wrapper.py
└── wrapper_utils.py
```

السبب: هذه الملفات مسؤولة عن CybORG، الترجمة، والـ observation graph. وهي مستقلة نسبياً عن نوع خوارزمية PPO.

---

## 3. الملفات التي يجب إضافتها

لإضافة MAPPO بدون كسر IPPO، لا تعدّل `ppo_agent.py` مباشرة في البداية.

الأفضل إضافة ملفات جديدة:

```text
src/models/agents/mappo_agent.py
src/models/gnn/centralized_critic.py
src/models/utils/global_state.py
src/trainers/mappo_trainer.py
src/trainers/mappo_collector.py
```

بهذا الشكل يبقى IPPO يعمل، وتضيف MAPPO تدريجياً.

---

## 4. إضافة Centralized Critic

### ملف جديد

```text
src/models/gnn/centralized_critic.py
```

### الدور

هذا الملف يحتوي Critic جديد يرى حالة مشتركة.

مثلاً:

```python
class CentralizedCriticNetwork(nn.Module):
    ...
```

### الفرق مع Critic الحالي

الـ Critic الحالي:

```text
InductiveCriticNetwork(local_graph)
```

الـ Critic الجديد:

```text
CentralizedCriticNetwork(global_graph)
```

---

## 5. بناء Global State

### ملف جديد

```text
src/models/utils/global_state.py
```

### الدور

هذا الملف يبني حالة مشتركة من ملاحظات جميع agents.

مثلاً:

```python
def build_global_state(agent_states):
    ...
```

المدخل:

```text
{
  blue_agent_0: obs0,
  blue_agent_1: obs1,
  blue_agent_2: obs2,
  blue_agent_3: obs3,
  blue_agent_4: obs4,
}
```

الخروج:

```text
global_state
```

يمكن أن يكون:

- graph مدمج
- features مدمجة
- أو تمثيل مبسط أولياً

---

## 6. تعديل Memory لدعم MAPPO

### الوضع الحالي

```text
src/models/memory/ppo_memory.py
src/models/memory/multi_ppo_memory.py
```

تخزن حالياً:

```text
state
action
value
log_prob
reward
terminal
```

### المطلوب في MAPPO

يجب تخزين:

```text
local_state
global_state
action
value
log_prob
reward
terminal
```

### خيار آمن

بدلاً من تعديل ملفات IPPO، أضف:

```text
src/models/memory/mappo_memory.py
```

ويحتوي:

```python
class MAPPOMemory:
    ...
```

أو:

```python
class MultiMAPPOMemory:
    ...
```

---

## 7. إضافة MAPPO Agent

### ملف جديد

```text
src/models/agents/mappo_agent.py
```

### الدور

يحتوي:

```python
class GraphMAPPOAgent:
    ...
```

### الفرق مع PPO Agent الحالي

في IPPO:

```python
value = self.critic(*local_state)
```

في MAPPO:

```python
value = self.critic(*global_state)
```

أما الـ Actor فيبقى:

```python
dist = self.actor(*local_state)
```

---

## 8. تعديل get_action

في IPPO الحالي:

```python
action, value, log_prob = agent.get_action(local_obs)
```

في MAPPO الأفضل:

```python
action, value, log_prob = agent.get_action(
    local_obs,
    global_state,
)
```

يعني:

```python
dist = self.actor(*local_obs)
value = self.critic(*global_state)
```

---

## 9. إضافة MAPPO Collector

### ملف جديد

```text
src/trainers/mappo_collector.py
```

### الدور

يشبه `collector.py` الحالي، لكن الفرق أنه يبني `global_state` في كل timestep.

في كل step:

```text
states
  ↓
build_global_state(states)
  ↓
for each agent:
    action = actor(local_state)
    value = centralized_critic(global_state)
```

ثم يخزن:

```text
local_state
global_state
action
value
log_prob
reward
terminal
```

---

## 10. إضافة MAPPO Trainer

### ملف جديد

```text
src/trainers/mappo_trainer.py
```

### الدور

يشبه `trainer.py` الحالي، لكنه يستعمل:

```text
GraphMAPPOAgent
mappo_collector.py
CentralizedCriticNetwork
MAPPOMemory
```

ويحتوي:

```python
def run_mappo_train(cfg):
    ...
```

---

## 11. تعديل builders.py

### الملف

```text
src/trainers/builders.py
```

### الوضع الحالي

يبني IPPO agents:

```python
build_agents(cfg)
```

### المطلوب

إضافة دالة جديدة:

```python
def build_mappo_agents(cfg):
    ...
```

ولا تحذف `build_agents`.

مثال:

```python
def build_mappo_agents(cfg):
    return [
        GraphMAPPOAgent(...)
        for _ in range(5)
    ]
```

---

## 12. تعديل config

### الملف

```text
configs/base.yaml
```

أضف:

```yaml
algorithm:
  name: ippo
```

ثم عند استعمال MAPPO:

```yaml
algorithm:
  name: mappo
```

أو في `configs/train.yaml`:

```yaml
algorithm:
  name: mappo
```

---

## 13. تعديل main.py

### الملف

```text
src/main.py
```

### الفكرة

بدلاً من تشغيل `run_train(cfg)` دائماً، اجعل الاختيار حسب الخوارزمية:

```python
if cfg.get("algorithm", {}).get("name") == "mappo":
    run_mappo_train(cfg)
else:
    run_train(cfg)
```

---

## 14. تعديل __init__.py

يجب تحديث:

```text
src/models/agents/__init__.py
src/models/gnn/__init__.py
src/trainers/__init__.py
```

### مثال

في:

```text
src/trainers/__init__.py
```

أضف:

```python
from .mappo_trainer import run_mappo_train
```

---

## 15. ترتيب التنفيذ المقترح

لا تغيّر كل شيء دفعة واحدة.

اتبع هذا الترتيب:

### Step 1

أضف:

```text
src/models/gnn/centralized_critic.py
```

واختبر:

```bash
python - <<'PY'
from src.models.gnn.centralized_critic import CentralizedCriticNetwork
print("Centralized Critic OK")
PY
```

---

### Step 2

أضف:

```text
src/models/utils/global_state.py
```

واختبر:

```bash
python - <<'PY'
from src.models.utils.global_state import build_global_state
print("Global State OK")
PY
```

---

### Step 3

أضف:

```text
src/models/memory/mappo_memory.py
```

واختبر import.

---

### Step 4

أضف:

```text
src/models/agents/mappo_agent.py
```

واختبر:

```bash
python - <<'PY'
from src.models.agents.mappo_agent import GraphMAPPOAgent
print("MAPPO Agent OK")
PY
```

---

### Step 5

أضف:

```text
src/trainers/mappo_collector.py
```

واختبر episode صغير.

---

### Step 6

أضف:

```text
src/trainers/mappo_trainer.py
```

واختبر import.

---

### Step 7

عدّل:

```text
src/main.py
```

لتشغيل MAPPO حسب config.

---

## 16. الفرق النهائي في Pipeline

### IPPO

```text
local_obs_i
   ↓
Actor_i
   ↓
action_i

local_obs_i
   ↓
Critic_i
   ↓
value_i
```

### MAPPO

```text
local_obs_i
   ↓
Actor_i
   ↓
action_i

global_state_all_agents
   ↓
Centralized Critic
   ↓
value_i أو value_global
```

---

## 17. الملفات النهائية بعد إضافة MAPPO

```text
src/models/
├── agents
│   ├── ppo_agent.py
│   └── mappo_agent.py
├── gnn
│   ├── actor.py
│   ├── critic.py
│   └── centralized_critic.py
├── memory
│   ├── ppo_memory.py
│   ├── multi_ppo_memory.py
│   └── mappo_memory.py
└── utils
    ├── graph_batching.py
    └── global_state.py

src/trainers/
├── collector.py
├── trainer.py
├── mappo_collector.py
└── mappo_trainer.py
```

---

## 18. الخلاصة

للانتقال من IPPO إلى MAPPO لا نغيّر البيئة ولا GraphWrapper.

التعديلات الأساسية تكون في:

```text
models/agents
models/gnn/critic
models/memory
models/utils/global_state
trainers/collector
trainers/trainer
main.py
configs
```

أفضل استراتيجية هي:

```text
لا تكسر IPPO
أضف MAPPO كمسار جديد
اختَر الخوارزمية من config
اختبر كل خطوة لوحدها
```

بهذا الشكل يمكن تشغيل:

```bash
python src/main.py train
```

مع IPPO أو MAPPO حسب:

```yaml
algorithm:
  name: ippo
```

أو:

```yaml
algorithm:
  name: mappo
```
