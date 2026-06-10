# IPPO → MAPPO : Guide de migration du projet

## Objectif

Le projet actuel utilise **IPPO (Independent PPO)**.

L'objectif est de migrer progressivement vers **MAPPO (Multi-Agent PPO)** tout en conservant l'architecture actuelle du projet.

---

# 1. Différence fondamentale entre IPPO et MAPPO

## IPPO

Chaque agent possède :

* son propre Actor
* son propre Critic

Le Critic ne voit que l'observation locale de l'agent.

```text
Agent 0 -> Actor(local_obs) + Critic(local_obs)
Agent 1 -> Actor(local_obs) + Critic(local_obs)
Agent 2 -> Actor(local_obs) + Critic(local_obs)
Agent 3 -> Actor(local_obs) + Critic(local_obs)
Agent 4 -> Actor(local_obs) + Critic(local_obs)
```

---

## MAPPO

L'Actor reste décentralisé.

Chaque agent choisit toujours son action avec sa propre observation.

```python
action = actor(local_observation)
```

En revanche le Critic devient centralisé.

```python
value = critic(global_state)
```

Le Critic voit :

* tous les agents
* tout le graphe
* tout l'état global

---

# 2. État actuel du projet

Aujourd'hui :

```text
src/models/agents/ppo_agent.py
```

contient :

```python
self.actor
self.critic
```

et dans :

```python
get_action()
```

on trouve :

```python
value = self.critic(*state)
```

Le Critic reçoit uniquement l'état local.

Nous sommes donc en IPPO.

---

# 3. Étape 1 : créer un Critic centralisé

Créer :

```text
src/models/gnn/centralized_critic.py
```

Ce fichier contiendra :

```python
class CentralizedCritic(nn.Module)
```

Son rôle :

* recevoir l'état global
* encoder le graphe global
* retourner une seule valeur V(s)

---

# 4. Étape 2 : construire un Global State

Aujourd'hui :

```python
ObservationGraph.get_state(agent)
```

retourne :

```python
agent_state
```

Il faut ajouter :

```python
ObservationGraph.get_global_state()
```

Cette fonction devra retourner :

```python
(
    x,
    edge_index,
    global_features
)
```

pour l'ensemble du réseau.

---

# 5. Étape 3 : modifier le Collector

Fichier :

```text
src/trainers/collector.py
```

Actuellement :

```python
action, value, log_prob = agent.get_action(obs)
```

Dans MAPPO :

```python
action = actor(local_obs)

value = critic(global_state)
```

Le Collector devra donc transmettre :

* local_state
* global_state

à chaque agent.

---

# 6. Étape 4 : modifier la mémoire PPO

Fichiers :

```text
src/models/memory/ppo_memory.py
src/models/memory/multi_ppo_memory.py
```

Aujourd'hui :

```python
state
action
reward
value
```

sont stockés.

MAPPO nécessite également :

```python
global_state
```

Ajouter par exemple :

```python
self.global_states = []
```

et enregistrer cet état dans :

```python
remember(...)
```

---

# 7. Étape 5 : modifier l'agent PPO

Fichier :

```text
src/models/agents/ppo_agent.py
```

Créer :

```text
src/models/agents/mappo_agent.py
```

Ne pas casser la version PPO existante.

Le nouvel agent devra contenir :

```python
self.actor
self.centralized_critic
```

---

# 8. Étape 6 : modifier learn()

Aujourd'hui :

```python
critic_values = self.critic(local_state)
```

MAPPO :

```python
critic_values = self.centralized_critic(global_state)
```

L'Actor continue à utiliser :

```python
local_state
```

Le Critic utilise :

```python
global_state
```

---

# 9. Étape 7 : ajouter un Builder MAPPO

Créer :

```text
src/trainers/builders_mappo.py
```

ou

```text
src/trainers/mappo_builder.py
```

avec :

```python
build_mappo_agents()
```

---

# 10. Étape 8 : ajouter un Trainer MAPPO

Créer :

```text
src/trainers/mappo_trainer.py
```

afin de conserver :

```text
trainer.py
```

pour IPPO.

---

# 11. Structure finale recommandée

```text
src/
│
├── models/
│   │
│   ├── agents/
│   │   ├── ppo_agent.py
│   │   └── mappo_agent.py
│   │
│   ├── gnn/
│   │   ├── critic.py
│   │   └── centralized_critic.py
│
├── trainers/
│   ├── trainer.py
│   ├── mappo_trainer.py
│   ├── builders.py
│   └── builders_mappo.py
│
├── observations/
│   └── graph/
│       └── observation_graph.py
```

---

# 12. Ordre conseillé de migration

Ne pas tout modifier d'un coup.

Ordre recommandé :

### Phase 1

Créer :

```text
centralized_critic.py
```

Tester les imports.

---

### Phase 2

Ajouter :

```python
get_global_state()
```

dans ObservationGraph.

Tester.

---

### Phase 3

Modifier :

```text
collector.py
```

pour récupérer le global state.

Tester.

---

### Phase 4

Modifier :

```text
memory/
```

pour stocker le global state.

Tester.

---

### Phase 5

Créer :

```text
mappo_agent.py
```

Tester.

---

### Phase 6

Créer :

```text
mappo_trainer.py
```

Tester.

---

### Phase 7

Ajouter :

```yaml
algorithm: mappo
```

dans les configurations YAML.

Permettre de choisir :

```bash
python src/main.py train
```

ou

```yaml
algorithm: ippo
```

ou

```yaml
algorithm: mappo
```

sans changer le reste du projet.

---

# Conclusion

Le passage IPPO → MAPPO nécessite principalement :

1. Un Critic centralisé.
2. Un état global partagé.
3. Une mémoire capable de stocker cet état global.
4. Un nouvel agent MAPPO.
5. Un nouveau pipeline d'entraînement.

L'Actor, les observations locales, l'environnement CAGE4, le GraphWrapper et la majorité de l'infrastructure actuelle peuvent être conservés.
