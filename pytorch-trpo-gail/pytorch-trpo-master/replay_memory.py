import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))
Transition_discriminator = namedtuple('Transition', ('state', 'action'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        # self.memory = []

        self.memory.append(Transition(*args))
        #print(len(self.memory))

    def sample(self):
        print(len(self.memory))
        return Transition(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)


class Memory_Discriminator(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition_discriminator(*args))

    def sample(self):
        return Transition_discriminator(*zip(*self.memory))

    def __len__(self):
        return len(self.memory)
