import numpy as np
import copy
from mcts.mcts_utils import StateNode, StateActionNode, print_tree

class Mcts(object):
    """
    Monte Carlo Tree Search Planning Method
    Normal version
    For finite action space and finite state space

    """
    def __init__(self, env, exploration_parameter, max_horizon, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.default_policy_fn = default_policy_fn
        self.env = env
        self.model_fn = model_fn
        self.select_action = self.select_action_uct
        self.max_horizon = max_horizon


        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    def action_1d_nd(self, action_1d, action_1d_dim, action_dim):
        # convert a number in 1d space into an vector in nd space 
        
        action_nd = [0]*action_dim
        res = copy.copy(action_1d)
        for d in reversed(range(action_dim)):

            divider = action_1d_dim**d

            action = res//divider
            
            res -= action * divider

            action_nd[d] = action

        return action_nd

    def run(self, st, rollout_times, debug=False):
        root_node = StateNode(st, parent=None, depth=0, reward=0, done=False)

        # grow the tree for N times
        for t in range(rollout_times):
            self.grow_tree(root_node)

        action = self.best_action(root_node)
        action_nd = self.action_1d_nd(action, self.ACTION_1D_DIM, self.ACTION_DIM)

        if debug:
            print_tree(root_node, depth=1)


        self.env.load_states(root_node.state)

        root_value = root_node.value()

        return action_nd, root_value

    def best_action(self, node):
        if node.num_children() == 0:
            print("no child in root node")
            action = self.default_policy_fn.get_action_1d(node.state)
        else:
            qs = []
            acs = []
            for child in node.children:

                q = child.value()
                qs.append(q)
                acs.append(child.action)
            qs = np.asarray(qs, dtype=np.float) + 1e-5
            qs[qs!=np.max(qs)] = 0

            logits = qs/np.sum(qs) # normalize
            best_q_idx = np.random.choice(node.num_children(), p=logits)
            action = acs[best_q_idx]

        return action

    def select_action_random(self, state_node):

        action = np.random.randint(self.env.ACTION_DIM)

        return action

    def select_action_uct(self, state_node):
        # select action according to uct value
        best_action = self.default_policy_fn.get_action_1d(state_node.state)
        best_q = -np.inf

        for action in range(self.env.ACTION_DIM):
            sa_node, exist = state_node.find_child(action)
            if not exist:
                value = np.inf
            else:
                # print("value", sa_node.value())
                # print("cp", self.cp * np.sqrt(np.log(state_node.visited_times)/sa_node.visited_times))
                value = sa_node.value() + self.cp * np.sqrt(np.log(state_node.visited_times)/sa_node.visited_times)

            if value > best_q:
                best_action = action
                best_q = value

        return best_action

    def aggregate_sa_node(self, state_node, action):
        new_sa_node, exist = state_node.find_child(action)

        if not exist:
            new_sa_node = StateActionNode(state_node.state, action, parent=state_node, depth=state_node.depth+1)
            state_node.append_child(new_sa_node)

        return new_sa_node

    def expansion(self, leaf_state_node):
        action = self.default_policy_fn.get_action_1d(leaf_state_node.state)
        new_sa_node = StateActionNode(leaf_state_node.state, action, parent=leaf_state_node, depth=leaf_state_node.depth+1)
        leaf_state_node.append_child(new_sa_node)

        action_nd = self.action_1d_nd(action, self.ACTION_1D_DIM, self.ACTION_DIM)
        state_nxt, reward, done = self.model_fn.step(leaf_state_node.state, action_nd)
        new_s_node = StateNode(state_nxt, parent=new_sa_node, depth=new_sa_node.depth+1., reward=reward, done=done)
        new_sa_node.append_child(new_s_node)
        new_sa_node.reward += reward


        return new_s_node, done

    def select_outcome(self, sa_node):
        # normal version
        action_nd = self.action_1d_nd(sa_node.action, self.ACTION_1D_DIM, self.ACTION_DIM)
        state_nxt, reward, done = self.model_fn.step(sa_node.state,action_nd)
        decision_node, exist = sa_node.find_child(state_nxt)

        if not exist:
            decision_node = StateNode(state_nxt, parent=sa_node, depth=sa_node.depth+1, reward=reward, done=done)
            sa_node.append_child(decision_node)

        sa_node.reward += decision_node.reward

        return decision_node, done

    def back_propogation(self, current_s_node, cumulative_reward):

        # backward phase
        while True:
            current_s_node.visited_times += 1
            current_s_node.cumulative_reward += cumulative_reward


            if current_s_node.parent == None:
                break

            current_sa_node = current_s_node.parent
            current_sa_node.visited_times += 1
            cumulative_reward += current_sa_node.reward_mean()
            current_sa_node.cumulative_reward += cumulative_reward
            current_s_node = current_sa_node.parent

    def grow_tree(self, root_node):
        current_s_node = root_node

        # forward phase
        while True:

            # select action add a (s,a) node into tree
            action = self.select_action(current_s_node)
            new_sa_node = self.aggregate_sa_node(current_s_node, action)

            # model generate next state add a (s) node into tree
            current_s_node, done = self.select_outcome(new_sa_node)

            if current_s_node.depth > self.max_horizon:
                break

            if current_s_node.visited_times == 0 or current_s_node.num_children() == 0:
                if not done:
                    current_s_node, done = self.expansion(current_s_node)
                break

        if not done:
            cumulative_reward = self.eval(current_s_node)
        else:
            cumulative_reward = 0.

        self.back_propogation(current_s_node, cumulative_reward)

    def eval(self, current_s_node):
        horizon = 0
        cumulative_reward = 0

        while True:
            horizon += 1
            action = self.default_policy_fn.get_action_1d(current_s_node.state)
            action_nd = self.action_1d_nd(action, self.ACTION_1D_DIM, self.ACTION_DIM)
            state_nxt, reward, done = self.model_fn.step(current_s_node.state, action_nd)
            cumulative_reward += reward

            if done or horizon > self.max_horizon:
                break
        # print("cumulative_reward: ", cumulative_reward)
        return cumulative_reward

class MctsSpw(Mcts):
    """
    Monte Carlo Tree Search Planning Method

    Single Progressive Widening
    For infinite action space finite state space

    """
    def __init__(self, env, exploration_parameter, max_horizon, alpha, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.alpha = alpha
        self.default_policy_fn = default_policy_fn
        self.env = env
        self.model_fn = model_fn
        self.select_action = self.select_action_spw
        self.max_horizon = max_horizon


        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    def select_action_spw(self, state_node):
        # select action single progressive widening

        # print("state_node.visited_times)**alpha", (state_node.visited_times)**alpha)
        # print("state_node.num_children()", state_node.num_children())

        if (state_node.visited_times)**self.alpha > state_node.num_children():
            action = self.default_policy_fn.get_action_1d(state_node.state)
            new_sa_node, exist = state_node.find_child(action)

            if not exist:
                new_sa_node = StateActionNode(state_node.state, action, parent=state_node, depth=state_node.depth+1)
                state_node.append_child(new_sa_node)
        else:
            # action = self.select_action_uct(state_node)
            action = self.select_action_random(state_node)

        return action

class MctsDpw(MctsSpw):
    """
    Monte Carlo Tree Search Planning Method

    Double Progressive Widening
    For infinite action space finite state space

    """
    def __init__(self, env, exploration_parameter, max_horizon, alpha, beta, default_policy_fn, model_fn):
        self.cp = exploration_parameter
        self.alpha = alpha
        self.beta = beta
        self.default_policy_fn = default_policy_fn
        self.env = env
        self.model_fn = model_fn
        self.select_action = self.select_action_spw
        self.max_horizon = max_horizon


        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    def choose_decision_node(self, sa_node):
        logits = []
        decision_nodes = []
        for child in sa_node.children:
            logits.append(child.visited_times)
            decision_nodes.append(child)

        logits = np.asarray(logits, dtype=np.float)
        logits = logits/np.sum(logits) # normalize
        node_idx = np.random.choice(sa_node.num_children(), p=logits)

        decision_node = decision_nodes[node_idx]

        return decision_node

    def select_outcome(self, sa_node):
        # double progressive widening version
        if (sa_node.visited_times)**self.beta >= sa_node.num_children():

            action_nd = self.action_1d_nd(sa_node.action, self.ACTION_1D_DIM, self.ACTION_DIM)
            state_nxt, reward, done = self.model_fn.step(sa_node.state, action_nd)
            decision_node, exist = sa_node.find_child(state_nxt)

            if not exist:
                decision_node = StateNode(state_nxt, parent=sa_node, depth=sa_node.depth+1, reward=reward, done=done)
                sa_node.append_child(decision_node)

        else:
            decision_node = self.choose_decision_node(sa_node)
            done = decision_node.done

        sa_node.reward += decision_node.reward

        return decision_node, done


