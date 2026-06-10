from abc import ABC
from collections.abc import Iterable
from collections import OrderedDict

import numpy as np

class Node(ABC, object):
    '''
    Ordered dict of features. Keys match CybORG output
    values are initialized to None for enums, -1 for scalars
    '''
    feats: OrderedDict

    '''
    The dimension of each of the features defined in feats
    E.g. if feats is [BooleanEnum, scaler]
    dims would be [2, 1]
    '''
    dims: Iterable[int]


    labels: list

    '''
    Defining __eq__ and __hash__ as so allows us to create a set
    of nodes to avoid duplicates later on
    '''
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        else:
            return False

    def __hash__(self):
        return self.uuid

    def __init__(self, uuid: int, observation: dict):
        self.uuid = uuid
        self.parse_observation(observation)


    def parse_observation(self, obs: dict) -> None:
        '''
        Given the dictionary from an observation, update/set
        features (perhaps using the set_features method)

        By default, only pulls out what can be given in an
        observation, but we have some node types with additional
        features. This method should be extended to accomidate
        '''
        for k in self.feats.keys():
            if k in obs:
                self.feats[k] = obs[k]

    def get_features(self) -> np.array:
        '''
        Convert from all the enums to a fixed size vector

        Requires the following fields to be initialized:
            self.dim: The output dimension of the feature vector
            self.dims: The output dimensions of individual one-hot features (sum(dims) == dim)
            self.feats: An ordered dict of the features we want
        '''
        out = np.zeros(self.dim)

        offset = 0
        for i,feat in enumerate(self.feats.values()):
            # Enums are 1-indexed
            if feat:
                if isinstance(feat, Iterable):
                    for f in feat:
                        out[offset + (f.value-1)] = 1
                elif isinstance(feat, float):
                    out[offset] = feat
                elif isinstance(feat, bool):
                    out[offset] = float(feat)
                else:
                    out[offset + (feat.value-1)] = 1

            offset += self.dims[i]

        return out

    def human_readable(self):
        human_readable = []
        for i,feat_str in enumerate(self.feats.keys()):
            if self.dims[i] > 1:
                human_readable += [f'{feat_str}-{j}' for j in range(self.dims[i])]
            else:
                human_readable.append(feat_str)

        return human_readable
