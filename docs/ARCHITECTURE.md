# Architecture du projet Adversarial MARL Cyber

Ce projet est organisé autour d’un pipeline complet pour entraîner, évaluer et sauvegarder des agents PPO multi-agents dans l’environnement CybORG/CAGE4. L’architecture sépare clairement les responsabilités : configuration, environnement, observation graphe, modèles GNN/PPO, entraînement, checkpoints et tests.

```
.
├── checkpoints
│   ├── model-0_checkpoint.pt
│   ├── model-1_checkpoint.pt
│   ├── model-2_checkpoint.pt
│   ├── model-3_checkpoint.pt
│   └── model-4_checkpoint.pt
├── configs
│   ├── base.yaml
│   ├── eval.yaml
│   └── train.yaml
├── docs
│   └── ARCHITECTURE.md
├── experiments
├── logs
│   └── model.pt
├── src
│   ├── __init__.py
│   ├── environments
│   │   ├── __init__.py
│   │   └── cage4
│   │       ├── __init__.py
│   │       ├── action_space.py
│   │       ├── action_translator.py
│   │       ├── constants.py
│   │       ├── env_factory.py
│   │       ├── graph_wrapper.py
│   │       ├── topology.py
│   │       └── wrapper_utils.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   └── ppo_agent.py
│   │   ├── gnn
│   │   │   ├── __init__.py
│   │   │   ├── actor.py
│   │   │   ├── attention.py
│   │   │   ├── critic.py
│   │   │   ├── encoders.py
│   │   │   └── helpers.py
│   │   ├── load.py
│   │   ├── memory
│   │   │   ├── __init__.py
│   │   │   ├── multi_ppo_memory.py
│   │   │   └── ppo_memory.py
│   │   └── utils
│   │       ├── __init__.py
│   │       └── graph_batching.py
│   ├── observations
│   │   ├── __init__.py
│   │   └── graph
│   │       ├── __init__.py
│   │       ├── graph_updates.py
│   │       ├── node_tracker.py
│   │       ├── nodes
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   ├── connection_node.py
│   │       │   ├── decoys.py
│   │       │   ├── file_node.py
│   │       │   ├── internet_node.py
│   │       │   └── system_node.py
│   │       └── observation_graph.py
│   ├── trainers
│   │   ├── __init__.py
│   │   ├── builders.py
│   │   ├── checkpoint.py
│   │   ├── collector.py
│   │   ├── config.py
│   │   ├── tpu.py
│   │   ├── train_loop.py
│   │   ├── trainer.py
│   │   └── updater.py
│   └── utils
│       ├── __init__.py
│       └── config_loader.py
└── tests
    ├── test_encoder.py
    ├── test_graph.py
    └── test_wrapper_import.py

```

---

## 1. Vue générale du flux

Le flux principal est le suivant :

```text
configs/
   ↓
src/main.py
   ↓
src/trainers/
   ↓
src/environments/cage4/
   ↓
src/observations/graph/
   ↓
src/models/
   ↓
checkpoints/ + logs/
```

Le système fonctionne ainsi :

1. `src/main.py` charge la configuration YAML.
2. Le trainer initialise les agents PPO et les environnements CAGE4.
3. `GraphWrapper` transforme les observations CybORG en observations graphes.
4. `ObservationGraph` maintient une représentation dynamique du réseau.
5. Les modèles GNN produisent les actions des agents.
6. Le collector exécute des épisodes et remplit les buffers PPO.
7. La boucle d’entraînement met à jour les réseaux Actor/Critic.
8. Les checkpoints et logs sont sauvegardés.

---

## 2. Dossiers racine

### `checkpoints/`

Contient les poids sauvegardés des agents entraînés.

Exemple :

```text
model-0_checkpoint.pt
model-1_checkpoint.pt
model-2_checkpoint.pt
model-3_checkpoint.pt
model-4_checkpoint.pt
```

Chaque fichier correspond à un agent bleu différent.

---

### `configs/`

Contient les fichiers YAML de configuration.

#### `configs/base.yaml`

Configuration principale du projet.

Elle définit :

* les chemins des logs et checkpoints
* les paramètres runtime
* les paramètres d’entraînement
* les hyperparamètres du modèle
* les paramètres d’évaluation

#### `configs/train.yaml`

Surcharge spécifique pour l’entraînement.

Exemple :

```yaml
run:
  name: model
```

#### `configs/eval.yaml`

Configuration dédiée à l’évaluation.

Elle définit par exemple :

* nombre d’épisodes d’évaluation
* seed
* dossier de sortie
* mode distribué ou non

---

### `docs/`

Contient la documentation du projet.

#### `docs/ARCHITECTURE.md`

Document qui décrit l’architecture générale, les responsabilités des dossiers et le flux complet du système.

---

### `experiments/`

Dossier prévu pour stocker des expériences, configurations alternatives ou résultats de runs spécifiques.

---

### `logs/`

Contient les logs d’entraînement.

Exemple :

```text
model.pt
```

Ce fichier peut stocker l’évolution des rewards et losses pendant l’entraînement.

---

## 3. Dossier `src/`

Le dossier `src/` contient tout le code source principal du projet.

---

## 4. `src/main.py`

Point d’entrée principal du projet.

Rôle :

* lire la commande utilisateur : `train` ou `eval`
* charger les fichiers YAML depuis `configs/`
* appeler la fonction d’entraînement ou d’évaluation

Exemple d’exécution :

```bash
python src/main.py train
```

ou :

```bash
PYTHONPATH=. python -m src.main train
```

---

## 5. `src/environments/`

Ce dossier contient tout ce qui concerne les environnements.

### `src/environments/__init__.py`

Expose les éléments principaux de l’environnement, par exemple :

* `GraphWrapper`
* `translate_action`
* `MAX_ACTIONS`
* `N_AGENTS`

---

## 6. `src/environments/cage4/`

Ce dossier contient l’intégration spécifique avec CybORG/CAGE4.

### `constants.py`

Contient les constantes globales de l’environnement CAGE4 :

* `N_AGENTS`
* `MAX_SERVERS`
* `MAX_USERS`
* `MAX_HOSTS`
* `SN_BLOCK_SIZE`
* `POSSIBLE_NEIGHBORS`

Ces constantes sont utilisées par le wrapper, l’action space et les modèles.

---

### `topology.py`

Décrit la topologie réseau CAGE4.

Contient :

* `ROUTERS`
* `ACCESSABLE_OFFLINE`
* `MY_SUBNETS`

`MY_SUBNETS` indique quels sous-réseaux sont contrôlés par chaque agent bleu.

---

### `action_space.py`

Définit l’espace d’actions disponible.

Contient :

* `NODE_ACTIONS`
* `EDGE_ACTIONS`
* `GLOBAL_ACTIONS`
* `N_NODE_ACTIONS`
* `N_EDGE_ACTIONS`
* `N_GLOBAL_ACTIONS`
* `MAX_ACTIONS`

Les actions incluent par exemple :

* Analyse
* Remove
* Restore
* DeployDecoy
* Monitor
* AllowTrafficZone
* BlockTrafficZone

---

### `action_translator.py`

Convertit une action discrète produite par le modèle PPO en action CybORG réelle.

Rôle :

```text
action_id entier
   ↓
action CybORG
```

Exemple :

```python
translate_action("blue_agent_0", 15)
```

Cette fonction choisit :

* le sous-réseau cible
* le type d’action
* l’hôte cible ou le lien réseau cible

---

### `env_factory.py`

Crée l’environnement CAGE4 complet.

Rôle :

* construire `EnterpriseScenarioGenerator`
* créer `CybORG`
* appliquer `GraphWrapper`

Fonction principale :

```python
make_env(seed=None, steps=500)
```

Elle retourne un environnement prêt pour l’entraînement.

---

### `graph_wrapper.py`

Wrapper principal autour de CybORG.

Rôle :

* recevoir les actions des agents
* les traduire en actions CybORG
* appeler `env.step`
* récupérer les observations brutes
* mettre à jour les graphes d’observation
* retourner une observation compatible avec le GNN

Fonctions principales :

* `reset()`
* `step(actions)`

Ce fichier est le lien central entre CybORG et le système GNN/PPO.

---

### `wrapper_utils.py`

Contient les fonctions utilitaires utilisées par `GraphWrapper`.

Fonctions principales :

* `parse_tabular()`
* `combine_data()`
* `to_obs()`

Rôle :

* extraire les features tabulaires
* concaténer features graphe + features tabulaires
* construire les tenseurs finaux utilisés par les réseaux GNN

---

## 7. `src/observations/`

Ce dossier contient toute la logique de représentation des observations.

### `src/observations/__init__.py`

Expose les composants principaux :

* `ObservationGraph`
* `NodeTracker`
* `GraphUpdatesMixin`

---

## 8. `src/observations/graph/`

Contient la représentation dynamique du réseau sous forme de graphe.

### `observation_graph.py`

Fichier principal de construction du graphe.

Classe principale :

```python
ObservationGraph
```

Rôle :

* construire la topologie initiale
* stocker les nœuds
* stocker les edges permanents
* stocker les edges temporaires
* générer les features du graphe
* générer les masks d’actions

Méthodes principales :

* `setup(initial_observation)`
* `parse_initial_observation(obs)`
* `get_state(subnets)`
* `set_firewall_rules(src, dst)`

---

### `graph_updates.py`

Contient la logique de mise à jour dynamique du graphe après les actions et observations.

Classe principale :

```python
GraphUpdatesMixin
```

Méthode principale :

```python
parse_observation(obs)
```

Elle gère notamment :

* Restore
* Remove
* DeployDecoy
* connexions réseau
* fichiers suspects
* processus observés

---

### `node_tracker.py`

Gère les identifiants numériques des nœuds.

Classe principale :

```python
NodeTracker
```

Rôle :

* associer un nom de nœud à un identifiant entier
* retrouver le nom depuis l’identifiant
* créer automatiquement de nouveaux IDs

---

## 9. `src/observations/graph/nodes/`

Contient les différents types de nœuds du graphe.

### `base.py`

Classe abstraite de base :

```python
Node
```

Rôle :

* stocker les features
* parser une observation
* convertir les features en vecteur numérique

---

### `system_node.py`

Définit :

```python
SystemNode
```

Représente :

* serveurs
* utilisateurs
* routeurs

Features possibles :

* architecture
* OS
* version
* patches
* rôle serveur/utilisateur/routeur

---

### `connection_node.py`

Définit :

```python
ConnectionNode
```

Représente :

* processus
* ports ouverts
* connexions
* services par défaut
* services éphémères
* decoys

---

### `file_node.py`

Définit :

```python
FileNode
```

Représente les fichiers observés sur les hôtes.

Features possibles :

* type de fichier
* chemin
* permissions
* version
* vendor
* signature
* densité

---

### `internet_node.py`

Définit :

```python
InternetNode
```

Nœud structurel représentant Internet.

---

### `decoys.py`

Contient :

```python
init_decoy()
```

Crée des `ConnectionNode` correspondant aux leurres déployés :

* apache2
* tomcat
* vsftpd
* haraka

---

## 10. `src/models/`

Contient toute la logique des modèles d’apprentissage.

### `src/models/__init__.py`

Expose les principaux composants du package models :

* agents PPO
* actor
* critic
* memory
* helpers
* loader

---

### `src/models/load.py`

Expose la fonction :

```python
load(path)
```

Elle charge un checkpoint et retourne un agent prêt pour l’inférence.

---

## 11. `src/models/agents/`

Contient les agents d’apprentissage.

### `ppo_agent.py`

Classe principale :

```python
InductiveGraphPPOAgent
```

Rôle :

* contenir l’Actor
* contenir le Critic
* gérer la mémoire PPO
* choisir une action
* apprendre avec PPO
* sauvegarder/charger les poids

Méthodes principales :

* `get_action(obs)`
* `remember(...)`
* `learn()`
* `save(path)`
* `load_weights(path)`
* `train()`
* `eval()`

---

## 12. `src/models/gnn/`

Contient les réseaux de neurones graphes.

### `actor.py`

Classe principale :

```python
InductiveActorNetwork
```

Rôle :

* encoder le graphe
* produire une distribution d’actions
* retourner une distribution `Categorical`

Il produit :

* actions sur les nœuds
* actions sur les edges
* actions globales

---

### `critic.py`

Classe principale :

```python
InductiveCriticNetwork
```

Rôle :

* encoder l’état graphe
* produire une valeur scalaire
* estimer la valeur de l’état pour PPO

---

### `encoders.py`

Contient :

```python
GraphEncoder
```

Encodeur GCN partagé.

Rôle :

* appliquer deux couches GCN
* retourner les représentations intermédiaires

---

### `attention.py`

Contient :

```python
SimpleSelfAttention
```

Rôle :

* agréger les informations des nœuds
* produire un vecteur global
* enrichir l’état global utilisé par Actor et Critic

---

### `helpers.py`

Contient :

* `pad_sequence()`
* `extract_hosts()`

Rôle :

* extraire les serveurs et utilisateurs du graphe
* construire des tenseurs batchés avec padding
* créer les masks nécessaires

---

## 13. `src/models/memory/`

Contient les buffers PPO.

### `ppo_memory.py`

Classe :

```python
PPOMemory
```

Stocke :

* états
* actions
* valeurs critic
* log-probabilités
* rewards
* terminaux

---

### `multi_ppo_memory.py`

Classe :

```python
MultiPPOMemory
```

Gère un buffer PPO par agent.

Utilisé pendant l’entraînement multi-agent.

---

## 14. `src/models/utils/`

Contient les utilitaires liés aux graphes.

### `graph_batching.py`

Fonctions :

* `combine_subgraphs()`
* `combine_marl_states()`

Rôle :

* concaténer plusieurs graphes
* décaler correctement les indices des edges
* construire un batch global pour Actor/Critic

---

## 15. `src/trainers/`

Contient toute la logique d’entraînement.

### `config.py`

Contient :

* `default_config()`
* `build_hyper_params()`

Rôle :

* fournir une configuration par défaut
* convertir la config en objet simple utilisé par la boucle de train

---

### `builders.py`

Contient :

* `build_agents()`
* `build_envs()`

Rôle :

* créer les 5 agents PPO
* créer les environnements CAGE4 via `make_env()`

---

### `collector.py`

Contient :

* `generate_episode()`
* `collect_data()`

Rôle :

* exécuter les épisodes
* collecter les transitions PPO
* accumuler les rewards
* remplir les buffers multi-agents

---

### `updater.py`

Contient :

```python
train_models()
```

Rôle :

* lancer `agent.learn()` pour chaque agent
* retourner les losses

---

### `checkpoint.py`

Contient :

* `save_checkpoints()`
* `load_checkpoints()`
* `save_logs()`
* `load_logs()`

Rôle :

* sauvegarder les modèles
* charger les checkpoints en mode resume
* sauvegarder les logs d’entraînement

---

### `tpu.py`

Contient :

* `start_tpu_test_async()`
* `check_tpu_test_async()`

Rôle :

* lancer un test TPU asynchrone
* vérifier la disponibilité TPU sans bloquer l’entraînement

---

### `train_loop.py`

Contient :

```python
train_loop()
```

Rôle :

* exécuter la boucle principale PPO
* collecter les rollouts
* entraîner les agents
* calculer reward/loss
* sauvegarder logs et checkpoints
* arrêter si le temps maximum est dépassé

---

### `trainer.py`

Contient :

```python
run_train()
```

Rôle :

* point d’entrée haut niveau du training
* charger les hyperparamètres
* construire agents et environnements
* charger checkpoints si nécessaire
* lancer `train_loop()`

---

## 16. `src/utils/`

Contient les outils généraux du projet.

### `config_loader.py`

Fonctions :

* `load_yaml()`
* `merge_dicts()`
* `set_nested_value()`
* `apply_overrides()`
* `load_config()`

Rôle :

* charger plusieurs YAML
* fusionner les configurations
* appliquer des overrides depuis la CLI

---

## 17. `tests/`

Contient les tests rapides du projet.

### `test_encoder.py`

Teste les composants GNN de base.

### `test_graph.py`

Teste la création du graphe d’observation.

### `test_wrapper_import.py`

Teste l’import du `GraphWrapper`.

---

## 18. Résumé des responsabilités

```text
configs/                 configuration YAML
src/main.py              point d’entrée CLI
src/environments/        création et wrapping de CybORG
src/observations/        transformation observation → graphe
src/models/              GNN, PPO, mémoire
src/trainers/            entraînement PPO
src/utils/               configuration et utilitaires globaux
checkpoints/             poids sauvegardés
logs/                    courbes et métriques d’entraînement
tests/                   tests unitaires simples
```

---

## 19. Résumé du pipeline d’entraînement

```text
python src/main.py train
        ↓
load_config()
        ↓
run_train()
        ↓
build_agents()
        ↓
build_envs()
        ↓
train_loop()
        ↓
collect_data()
        ↓
generate_episode()
        ↓
GraphWrapper.step()
        ↓
ObservationGraph.get_state()
        ↓
InductiveGraphPPOAgent.get_action()
        ↓
MultiPPOMemory
        ↓
agent.learn()
        ↓
save_checkpoints()
        ↓
save_logs()
```

Cette architecture sépare clairement l’ancien fichier monolithique du projet original en plusieurs modules spécialisés, ce qui rend le projet plus lisible, testable et maintenable.
