# src/environments/cage4/env_factory.py

from CybORG import CybORG

from CybORG.Agents import (
    SleepAgent,
    EnterpriseGreenAgent,
    FiniteStateRedAgent,
)

from CybORG.Simulator.Scenarios import (
    EnterpriseScenarioGenerator,
)

from .graph_wrapper import GraphWrapper


def make_env(
    seed=None,
    steps=500,
):
    """
    Build a CAGE4 environment wrapped
    with our graph observation wrapper.
    """

    scenario = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )

    env = CybORG(
        scenario,
        "sim",
        seed=seed,
    )

    return GraphWrapper(env)