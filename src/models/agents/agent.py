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
    def __init__(self, in_dim, a_kwargs=None, c_kwargs=None, training=True, concat_edges=False, device="cpu"):
        a_kwargs = a_kwargs or {}
        c_kwargs = c_kwargs or {}

        self.actor = InductiveActorNetwork(in_dim, concat_edges=concat_edges, **a_kwargs)
        self.critic = InductiveCriticNetwork(in_dim, **c_kwargs)
        self.device = device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.memory = None
        self.kwargs = None
        self.args = None

        self.training = training
        self.deterministic = False
    
    def to(self, device):
        self.device = device
        self.actor.to(device)
        self.critic.to(device)
        return self


    def _move_to_device(self, data):
        if torch.is_tensor(data):
            return data.to(self.device)

        if isinstance(data, tuple):
            return tuple(self._move_to_device(x) for x in data)

        if isinstance(data, list):
            return [self._move_to_device(x) for x in data]

        if isinstance(data, dict):
            return {k: self._move_to_device(v) for k, v in data.items()}

        return data

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

    def _zero_grad(self):
        '''
        Reset opt
        '''
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def _is_xla_device(self):
        return "xla" in str(self.device).lower()


    def _step(self):
        if self._is_xla_device():
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(self.actor.opt)
            xm.optimizer_step(self.critic.opt)
            xm.mark_step()
        else:
            self.actor.opt.step()
            self.critic.opt.step()


    def set_deterministic(self, val):
        self.deterministic = val

    def set_mems(self, mems):
        if self.memory is None:
            raise ValueError("Memory must be initialized before setting mems")
        self.memory.mems = mems

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

    @torch.no_grad()
    def get_action(self, obs, *args):
        '''
        Sample an action from the actor's distribution
        given the current state. 

        If eval(), only returns the action 
        If train() returns action, value, and log prob 
        '''
        state,is_blocked = obs
        if is_blocked:
            return None
        state = self._move_to_device(state)
        distro = self.actor(*state)

        # I don't know why this would ever be called
        # during training, but just in case, putting the
        # logic block outside the training check
        if self.deterministic:
            action = distro.probs.argmax()
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(*state)
        prob = distro.log_prob(action)
        return action.item(), value.item(), prob.item()

    def load_weights(self, path):
        data = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(data['actor'])
        self.critic.load_state_dict(data['critic'])
        self.to(self.device)

    @abstractmethod
    def remember(self, *args):
        pass

    @abstractmethod
    def learn(self, verbose=False):
        pass

