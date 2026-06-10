# src/environments/cage4/graph_wrapper.py

from copy import deepcopy

import numpy as np

from CybORG.env import CybORG
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from CybORG.Shared.Enums import TernaryEnum

from src.environments.cage4.topology import MY_SUBNETS

from src.observations import ObservationGraph
 

from .wrapper_utils import combine_data, parse_tabular, to_obs
from .action_translator import translate_action

class GraphWrapper(EnterpriseMAE):

    def __init__(self, env: CybORG, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.graphs = dict()
        self.env = env
        self.agent_names = [f'blue_agent_{i}' for i in range(5)]
        self.ts = 0

        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

   
    def step(self, action):
        '''
        Take an action, execute it, and update internal approximation of the
        environment based on new observation.

        Args: 
            action: dict of {agent_id (int) : action_id (int)}
        '''
        # Convert from model out to Action objects
        action = {
            k:translate_action(k,v)
            for k,v in action.items()
        }

        # Gets the info from the tabular wrapper (4 dims per host, in order)
        observation, reward, term, trunc, info = super().step(
            action_dict=action, messages=self.msg
        )

        # Tell ObservationGraph what happened and update
        graph_obs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = observation[agent]
            g = self.graphs[agent]

            # Get the raw observation dictionary for this agent
            dict_obs = self.env.environment_controller.get_last_observation(agent).data
            msg = dict_obs.pop('message')
            msg = np.stack(msg, axis=0)

            # Indicates if msg was recieved or comms are blocked
            # This way we differentiate between feature for 0 and unknown
            recieved_msg = msg[:, -1:]
            if i != 4:
                # Repeat agent 4's 'is_recieved' message across 2 more subnets
                recieved_msg = np.concatenate([recieved_msg, np.zeros((2,1))], axis=0)
                recieved_msg[-2:] = recieved_msg[-3]

                # Pull out messages for 'was_scanned' and 'was_comprimised'
                msg_small = msg[:-1, :2]
                msg_big = msg[-1, :6].reshape(3,2)
                msg = np.concatenate([msg_small, msg_big], axis=0)
            else:
                msg = msg[:, :2]

            msg = np.concatenate([msg, recieved_msg], axis=1)

            # Update the graph based on the raw dictionary 
            g.parse_observation(dict_obs)

            # Pull node features from tabular observation, and also update 
            # graph subnet connectivity edges. 
            tab_x,phase,new_msg = parse_tabular(o, g) 
            
            self.msg[agent] = new_msg

            # Combine node features from graph source, and tabular source
            x,ei,masks = g.get_state(MY_SUBNETS[i])
            x = combine_data(x, tab_x)

            # Mask/pack into conviniently sized tensors for the GNN models 
            obs = to_obs(x,ei,masks,phase,msg,new_msg)

            # During training, we need to know if the agent is still mid-action
            # If so, we don't bother calculating an action next turn 
            is_blocked = dict_obs['success'] == TernaryEnum.IN_PROGRESS
            graph_obs[agent] = (obs, is_blocked)

        self.ts += 1
        self.last_obs = graph_obs
        return graph_obs, reward, term, trunc, info

    def reset(self):
        '''
        Rebuild internal graph representation with parameters of new environment
        '''
        self.ts = 0

        obs_tab, action_mask = super().reset()
        g = ObservationGraph()

        # I don't *think* this is cheating, because FixedActionWrapper gets
        # to manipulate the obs returned by env.reset() which is the same thing.
        # Graph updates after intialization will all be using partial knowledge
        # known only to the agents.
        obs_dict = self.env.environment_controller.init_state
        g.setup(obs_dict)

        # Set message to empty for all agents
        self.msg = {
            a:np.zeros(8)
            for a in self.agent_names
        }

        my_state = dict()
        self.graphs = dict()
        for i in range(5):
            agent = f'blue_agent_{i}'
            o = obs_tab[agent]

            # Message from agent 4 has 2 extra subnet infos
            if i != 4:
                dummy_msg = (np.zeros((6,3)), np.zeros(8))
            else:
                dummy_msg = (np.zeros((4,3)), np.zeros(8))

            # Duplicate shared observation of the initial graph across
            # all agents (but make sure not to pass by reference)
            g_ = deepcopy(g)
            self.graphs[agent] = g_

            # Get tabular features and update connectivity graph 
            tab_x,phase,_ = parse_tabular(o,g_)

            # Combine all node features together and package for agents
            x,ei,masks = g_.get_state(MY_SUBNETS[int(agent[-1])])
            x = combine_data(x, tab_x)
            obs = to_obs(x,ei,masks,phase, *dummy_msg)

            # By default, agents are not blocked on turn 0
            my_state[agent] = (obs, False)

        self.last_obs = my_state
        return my_state, action_mask
