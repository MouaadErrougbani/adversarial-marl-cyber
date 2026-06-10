from src.observations.graph.observation_graph import (
    ObservationGraph,
)

graph = ObservationGraph()

print("Graph créé")

print(
    "Méthode parse_observation:",
    hasattr(graph, "parse_observation"),
)

print(
    "Nombre de nœuds:",
    len(graph.nodes),
)

print("Test terminé")