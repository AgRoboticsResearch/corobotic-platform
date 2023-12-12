import numpy as np
import copy

class MctsSparse(object):
    """
    Sparse sampling method
    https://www.cis.upenn.edu/~mkearns/papers/sparsesampling-journal.pdf

    Change a little bit, the reward at time step t also take consider as stochastic and put it into mean
    """
    def __init__(self, env, plan_depth, gamma, model_fn, default_policy_fn, extra_search_depth=0, extra_search_sample=1):
        self.env = env
        self.model_fn = model_fn

        self.PLAN_DEPTH = plan_depth
        self.GAMMA = gamma
        self.default_policy_fn = default_policy_fn

        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

        self.model_step_num = 0

        self.extra_search_depth = extra_search_depth
        self.extra_search_sample = extra_search_sample

    def __action_1d_nd(self, action_1d, action_1d_dim, action_dim):
        # convert a number in 1d space into an vector in nd space 
        
        action_nd = [0]*action_dim
        res = action_1d

        for d in reversed(range(action_dim)):

            divider = action_1d_dim**d

            action = res//divider
            
            res -= action * divider

            action_nd[d] = action

        return action_nd

    def __func_a(self, plan_depth, sample_num_per_action, gamma, state_t):
        q_as = self.__func_q(plan_depth, sample_num_per_action, gamma, state_t)
        best_action = np.argmax(q_as)
        best_q = q_as[best_action]
        
        # print("q_as: ", q_as)
        # print("best_action_idx: ", best_action)

        
        best_action = self.__action_1d_nd(best_action, self.ACTION_1D_DIM, self.ACTION_DIM)
        # print("best_action: ", best_action)
        return best_action, best_q

    def __random_evaluate(self, state_t, eval_depth, eval_times):

        q_a = 0

        for n in range(eval_times):
            value = 0
            depth = 0
            state_t_internal = copy.copy(state_t)
            while True:
                depth += 1
                if depth > eval_depth:
                    break
                action = self.default_policy_fn.get_action(state_t_internal)
                state_nxt, num_picked_t, reach_end = self.model_fn.step(state_t_internal, action)
                state_t_internal = state_nxt
                value += num_picked_t
                if reach_end:
                    break
            q_a += value
        q_a /= eval_times

        return q_a



    def __func_q(self, plan_depth, sample_num_per_action, gamma, state_t):

        # if plan_depth == 0:

        #     q_as = np.zeros(self.ACTION_DIM_FLAT) 

        #     # --------------- comment it out if original sparse samping not random evaluate at end
        #     if self.extra_search_depth > 0:
        #         for a in range(self.ACTION_DIM_FLAT):
        #             action = self.__action_1d_nd(a, self.ACTION_1D_DIM, self.ACTION_DIM)
        #             state_nxt, num_picked_t, reach_end = self.model_fn.step(state_t, action)
        #             q_a = num_picked_t + self.__random_evaluate(state_nxt, eval_depth=self.extra_search_depth, eval_times=self.extra_search_sample)
        #             q_as[a] = q_a
        #     # ------------------------------------------------------------------------------------

        #     return q_as

        q_as = []

        # print('plan_depth', plan_depth)

        for a in range(self.ACTION_DIM_FLAT):
            action = self.__action_1d_nd(a, self.ACTION_1D_DIM, self.ACTION_DIM)
            sample_values = []
            num_pickeds = []
            for n in range(sample_num_per_action):

                state_nxt, num_picked_t, reach_end = self.model_fn.step(state_t, action)
                # print(state_nxt)
                self.model_step_num += 1

                if reach_end:
                    value_nxt = 0.
                else:
                    value_nxt = self.__func_v(plan_depth-1, sample_num_per_action, gamma, state_nxt)

                sample_value = num_picked_t + gamma * value_nxt
                sample_values.append(sample_value)

            # print("sample_values", sample_values)
            q_a = np.mean(sample_values)

            # print("num_picked_t", num_picked_t)

            # print("small a", a, ' q ',q_a)
            # print('action', action)
            q_as.append(q_a)

        return q_as

    def __func_v(self, plan_depth, sample_num_per_action, gamma, state_t):

        if plan_depth == 0:
            if self.extra_search_depth == 0 or self.extra_search_sample == 0:
                return 0
            else:
                v = self.__random_evaluate(state_t, eval_depth=self.extra_search_depth, eval_times=self.extra_search_sample)
                return v
        else:
            q_as = self.__func_q(plan_depth, sample_num_per_action, gamma, state_t)
            return np.max(q_as)

    def run(self, sample_num_per_action):


        state_t = self.env.save_states()
        # print("run1")
        # print(state_t)
        
        # monte carlo sampling
        action, best_q = self.__func_a(self.PLAN_DEPTH, sample_num_per_action, self.GAMMA, state_t)

        # load back states    
        self.env.load_states(state_t)
        # print("run2")
        # print(state_t)




        return action, best_q

