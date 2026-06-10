#src/environments/cage4/action_translator.py

from src.environments.cage4.action_space import (
    MAX_ACTIONS,
    NODE_ACTIONS,
    EDGE_ACTIONS,
    N_NODE_ACTIONS,
    N_EDGE_ACTIONS,
)

from src.environments.cage4.topology import (
    MY_SUBNETS,
    ROUTERS,
)

from src.environments.cage4.constants import (
    MAX_HOSTS,
    POSSIBLE_NEIGHBORS,
)

from src.environments.cage4.action_space import Monitor

def translate_action(agent_name, a_id):
        '''
        Translates output of PPO model to an action for the CybORG env. 
        Model provides output as
        Node-actions, edge-actions, global-actions, per-subnet.
        '''
        session = 0 # Seems the same every time?

        if a_id is None:
            return Monitor(session, agent_name)

        agent_id = int(agent_name[-1])
        which_subnet = MY_SUBNETS[agent_id][a_id // MAX_ACTIONS]
        a_id %= MAX_ACTIONS

        # Node action
        if a_id < N_NODE_ACTIONS*MAX_HOSTS:
            a = NODE_ACTIONS[a_id // MAX_HOSTS]
            target = a_id % MAX_HOSTS

            if target > 5:
                target = f'{which_subnet}_user_host_{target-6}'
            else:
                target = f'{which_subnet}_server_host_{target}'

            return a(session=session, agent=agent_name, hostname=target)

        # Edge action
        elif (a_id := a_id - (N_NODE_ACTIONS*MAX_HOSTS)) < (N_EDGE_ACTIONS*POSSIBLE_NEIGHBORS):
            a = EDGE_ACTIONS[a_id // POSSIBLE_NEIGHBORS]
            target = [r for r in ROUTERS if which_subnet not in r][a_id % POSSIBLE_NEIGHBORS]

            return a(session, agent_name, target.replace('_router',''), which_subnet)

        # Global action (only one)
        else:
            return Monitor(session, agent_name)
