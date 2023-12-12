import numpy as np
from queue import Queue


class StateNode(object):
    def __init__(self, state, parent, depth, reward, done):
        self.type = "state_node"
        self.state = state
        self.parent = parent
        self.children = []
        self.visited_times = 0
        self.cumulative_reward = 0.
        self.reward = reward
        self.depth = depth
        self.done = done

    def find_child(self, action):
        # check if this child already exist
        # for determinisitc and stochasitc action
        
        exist = False
        exist_child = None
        for child in self.children:
            if child.action == action:
                exist = True
                exist_child = child

        return exist_child, exist

    def append_child(self, child_node):
        self.children.append(child_node)

    def value(self):
        value = self.cumulative_reward / self.visited_times
        return value

    def num_children(self):
        num_children = len(self.children)
        return num_children

class StateActionNode(object):
    def __init__(self, state, action, parent, depth):
        self.type = "state_action_node"
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visited_times = 0
        self.cumulative_reward = 0.
        self.reward = 0.
        self.depth = depth

    def compare_states(self, states1, states2):
        same = True
        for i, state in enumerate(states1):
            if np.ndim(state != states2[i]) > 0:
                if (state != states2[i]).any():
                    same = False
            else:
                if (state != states2[i]):
                    same = False
        return same

    def find_child(self, state_nxt, compare=True):
        # check if this child already exist
        # 1. if compare flag is false it only work for determinisitc model, but faster.

        exist = False
        exist_child = None

        if compare:
            for child in self.children:
                # print("same: ", self.compare_states(child.state, state_nxt))
                if self.compare_states(child.state, state_nxt):
                    exist = True
                    exist_child = child
        else:
            if self.num_children() != 0:
                exist = True
                exist_child = self.children[0]

        return exist_child, exist

    def append_child(self, child_node):
        self.children.append(child_node)

    def value(self):
        value = self.cumulative_reward / self.visited_times
        return value

    def reward_mean(self):
        reward_mean = self.reward / self.visited_times
        return reward_mean

    def num_children(self):
        num_children = len(self.children)
        return num_children

def print_tree(node, depth=0):
    open_set = Queue()
    closed_set = set()

    root_node = node
    open_set.put(node)
    while not open_set.empty():
        cr_node = open_set.get()

        if depth > 0 and cr_node.depth > depth:
            break

        print_node_info(cr_node)

        for child in cr_node.children:
            if child in closed_set:
                continue
            # if child not in open_set:
            open_set.put(child)
            
        closed_set.add(cr_node)

def print_node_info(node):
    print("")
    print("depth: %i " %node.depth)
    print("type: ", node.type)
    if node.type == "state_action_node":    
        print("action: ", node.action)
    print("visited_times: ", node.visited_times)
    print("cumulative_reward: ", node.cumulative_reward)
    print("num_children: ", node.num_children())
    print("value: ", node.value())


class ModelWrapper(object):
    # wrap env model to fit mcts 
    def __init__(self, env):
        self.env = env
        
    def step(self, state, action):
        state_nxt, num_picked, reach_end = self.env.model(state, action)

        return state_nxt, num_picked, reach_end



