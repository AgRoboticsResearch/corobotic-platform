import numpy as np

from queue import *

class ModelWrapper(object):
    # wrap env model to fit mcts 
    def __init__(self, env):
        self.env = env
        
    def step(self, state, action):

        state_nxt, num_picked, reach_end = self.env.model(state, action)

        return state_nxt, num_picked, reach_end





