# src/models/agents/agent.py
import torch

from src.models.gnn.actor import (
    InductiveActorNetwork,
)

from src.models.gnn.critic import (
    InductiveCriticNetwork,
)



from abc import ABC, abstractmethod

class InductiveGraphAgent(ABC):
    '''
    Class base to manage agents' memories and learning (when training)
    When training is complete, uses the InductiveActorNetwork to decide
    which action to take
    '''
    def __init__(self, in_dim, a_kwargs=None, c_kwargs=None, training=True, concat_edges=False, critic = None, actor = None):
        a_kwargs = a_kwargs or {}
        c_kwargs = c_kwargs or {}
        
        self.actor = actor or InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.critic = critic or InductiveCriticNetwork(in_dim, **c_kwargs)
        self.external_actor  = None
        self.external_critic  = None
        self.memory = None
        self.kwargs = None
        self.args = None
        self._algorithm = None

        self.training = training
        self.deterministic = False
    




    # Required by CAGE but not utilized
    def end_episode(self):
        pass

    # Required by CAGE but not utilized
    def set_initial_values(self, action_space, observation):
        pass

    def train(self):
        '''
        Set modules to training mode
        '''
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        '''
        Set modules to eval mode 
        '''
        self.training = False
        self.actor.eval()
        self.critic.eval()
    
    def to(self, device):
        '''
        Move modules to device
        '''
        self.actor.to(device)
        self.critic.to(device)

    def _zero_grad(self):
        '''
        Reset opt
        '''
        if self.external_critic is None and self.external_actor is None:
            raise ValueError("Cannot zero grad when both actor and critic are external")
        if not self.external_actor :
            self.actor.opt.zero_grad()
        if not self.external_critic :
            self.critic.opt.zero_grad()


    def _step(self):
        if self.external_actor is None and self.external_critic is None:
            raise ValueError("Cannot step when both actor and critic are external")
        if not self.external_actor :
            self.actor.opt.step()
        if not self.external_critic :
           
            self.critic.opt.step()


    def set_deterministic(self, val):
        self.deterministic = val

    def set_memories(self, memories):
        if self.memory is None:
            raise ValueError("Memory must be initialized before setting memories")
        self.memory.memories = memories

    def save(self, path='saved_models/agent.pt'):
        if self.kwargs is None:
            raise ValueError("Agent must be initialized with kwargs to save")
        if self.args is None:
            raise ValueError("Agent must be initialized with args to save")
        me = (self.args, self.kwargs)

        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'agent': me
        }, path)
        
    def load_weights(self, path):
        data = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        

    @property
    def algorithm(self):
        if self._algorithm is None:
            raise ValueError(
                "Algorithm not set for this agent"
            )

        return self._algorithm

    @torch.no_grad()
    @abstractmethod
    def get_action(self, obs, *args):
        """
        Retourne une action à partir de l'observation courante.
        """
        pass
    
    @abstractmethod
    def remember(self, *args):
        pass

    @abstractmethod
    def learn(self, verbose=False):
        pass

