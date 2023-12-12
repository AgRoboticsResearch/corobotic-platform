from __future__ import print_function
from mcts.mcts_sparse import MctsSparse
from mcts.mcts import Mcts, MctsSpw, MctsDpw
from simulator.controller import RandomController
from simulator.appleSimulator import appleSimulator
from mcts.mcts_utils import ModelWrapper

import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import random
from multiprocessing import Pool

def generate_random_external_map(height, width, init_offset, density):
    # ---------------
    # params in: 
    # height, width: in pixels
    # init_offset: distance from platform x to edge of camera view in pixels

    external_map = np.random.randint(0, density, size=[height, width], dtype=np.int32)
    external_map[:, :init_offset] = 0
    return external_map

def plot_results(x, y, y_std, xlabel="x", ylabel="y", title="Title"):
    fig = plt.figure().gca()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cis = (y - y_std, y + y_std)
    fig.fill_between(x,cis[0],cis[1],alpha=0.2)
    fig.plot(x, y, 'ro')
    fig.margins(y=0)
    plt.show()

def mcts_dwp_experiment(appEnv, state_init, repeat, sample_num):
    results_nd = []
    model_fn = ModelWrapper(appEnv)

    for i in range(repeat):
        print("Repeating ... ", i)
        args = []
        for n in sample_num:
            state_t = copy.deepcopy(state_init)
            seed = i
            args.append([appEnv, state_t, n, seed,
                     model_fn, False, False])

        pool = Pool() # multi processing
        results = pool.map(eval_multi_helper_mcts_dwp, args)
        results_nd.append(results)

    pool.close()
    pool.join()
    results = np.asarray(results_nd)
    results_mean = np.mean(results, axis=0)
    results_sdt = np.std(results, axis=0)

    print("results ", results)
    print("results_mean ", results_mean)
    print("results_sdt ", results_sdt)
    return results, results_mean, results_sdt


def mcts_sparse_experiment(appEnv, state_init, repeat, plan_depth, sample_num_per_action, extra_search_depth):
    # repeat:  number
    # other inputs are lists   
    model_fn = ModelWrapper(appEnv)
    args = []

    for i in range(repeat):
        for d in plan_depth:
            for n in sample_num_per_action:
                for m in extra_search_depth:
                    state_t = copy.deepcopy(state_init)
                    seed = i
                    args.append([appEnv, state_t, d, n, seed, 
                             model_fn, m, False, False])

    pool = Pool() # multi processing
    results = pool.map(eval_multi_helper_mcts_sparse, args)
    pool.close()
    pool.join()
    
    results = np.asarray(results)
    results = results.reshape((repeat, -1))
    results_mean = np.mean(results, axis=0)
    results_sdt = np.std(results, axis=0)

    print("results ", results)
    print("results_mean ", results_mean)
    print("results_sdt ", results_sdt)
    return results, results_mean, results_sdt




def eval_multi_helper_mcts_dwp(args):
    # auxiliary funciton to make evaluating multiprocessing with multi arguments
    return evaluate_mcts_dpw(*args)

def eval_multi_helper_mcts_sparse(args):
    # auxiliary funciton to make evaluating multiprocessing with multi arguments
    return evaluate_mcts_sparse(*args)

def evaluate_random_policy(appEnv, state_t, model_fn, render=False, verbose=False, save=False):
    random_policy_fn = RandomController(appEnv)

    appEnv.load_states(state_t)
    num_apple_start = appEnv.count_apple()

    t_start = time.time()

    while True:
        if render:
            appEnv.show_map(save=save)

        action = random_policy_fn.get_action(state_t)
        state_nxt, num_picked, done = appEnv.model(state_t, action)
        state_t = state_nxt

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = appEnv.count_apple()
    picked_percent = (1 - num_apple_left/num_apple_start)*100

    if verbose:
        print("random")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent))
        # print("theoretic_max_pick: ", appEnv.theoretic_max_pick())
        # print("Theoretic pick percent: %.2f %%" %( (num_apple_start-num_apple_left)/appEnv.theoretic_max_pick()*100))

    return picked_percent

def evaluate_mcts_dpw(appEnv, state_t, rollout_times, random_seed, model_fn, render=False, verbose=False):
    if verbose:
        print("--------------------------------")
        print("Evaluating mcts dpw ......")
        print("sample_num: ", rollout_times)

    np.random.seed(random_seed) 
    random.seed(random_seed)

    # constant parameters
    cp = 10.
    alpha = 0.5
    beta = 0.5
    max_horizon = 10
    random_policy_fn = RandomController(appEnv)

    mcts = MctsDpw(appEnv, cp, max_horizon, alpha, beta, random_policy_fn, model_fn)

    picked_percents = []


    appEnv.load_states(state_t)
    num_apple_start = appEnv.count_apple()

    t_start = time.time()

    step = 0

    while True:
        if render:
            appEnv.show_map(save=save)
        action, best_q = mcts.run(st=state_t, rollout_times=rollout_times)
        # print("best action: ", action, "best_q: ", best_q)
        state_nxt, num_picked, done = appEnv.model(state_t, action)

        step += 1
        state_t = state_nxt

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = appEnv.count_apple()
    picked_percent = (1 - num_apple_left/num_apple_start)*100
    picked_percents.append(picked_percent)

    if verbose:
        print("Apple num: ", num_apple_start)
        print("Time", cost_time)
        print("Computation time per step %.2f s" %(cost_time/step))
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent))


    return picked_percent


def evaluate_mcts_sparse(appEnv, state_t, plan_depth, sample_num_per_action, random_seed, model_fn, extra_search_depth, render=False, verbose=False):
    if verbose:

        print("--------------------------------")
        print("Evaluating mcts sparse ......")
        print("sample num per action: ", sample_num_per_action)
        print("plan_depth: ", plan_depth)
        print("extra_search_depth: ", extra_search_depth)

    np.random.seed(random_seed) 
    random.seed(random_seed)

    # constant parameters
    gamma = 1.
    random_policy_fn = RandomController(appEnv)
    sparse_sampling = MctsSparse(appEnv, plan_depth, gamma, model_fn, random_policy_fn, extra_search_depth=extra_search_depth)

    picked_percents = []


    appEnv.load_states(state_t)
    num_apple_start = appEnv.count_apple()

    t_start = time.time()

    step = 0

    while True:
        if render:
            appEnv.show_map(save=save)
        action, best_q = sparse_sampling.run(sample_num_per_action=sample_num_per_action)
        # print("best action: ", action, "best_q: ", best_q)
        state_nxt, num_picked, done = appEnv.model(state_t, action, random_seed)

        step += 1
        state_t = state_nxt

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = appEnv.count_apple()
    picked_percent = (1 - num_apple_left/num_apple_start)*100
    picked_percents.append(picked_percent)

    if verbose:
        print("Apple num: ", num_apple_start)
        print("Time", cost_time)
        print("Computation time per step %.2f s" %(cost_time/step))
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent))

    return picked_percent

def evaluate_mcts_policy(appEnvs, repeat, pickers_x, pickers_speed, sample_num=10, plan_depth=1, render=False, verbose=False):
    print("--------------------------------")
    print("Evaluating mcmpc_policy ......", repeat, " repeat", len(appEnvs), " envs")

    print("sample_num: ", sample_num)
    print("plan_depth: ", plan_depth)

    tic = time.time()

    total_times = []

    for appEnv in appEnvs:
        ori_states = appEnv.save_states()

        for i in range(repeat):
            appEnv.load_states(ori_states)

            # init exp
            t = 0

            while True:
                pickers_x = appEnv.policy_mcts(sample_num=sample_num, plan_depth=plan_depth, verbose=verbose)
                reach_end, _ = appEnv.step(pickers_x)
                t += 1
                if reach_end:
                    total_times.append(t * appEnv.time_step)
                    if verbose:
                        print("Experiment: ", i)
                        print("reach_end")
                        print("total steps:", t)
                        print("total time:", t * appEnv.time_step, " s")
                    break
                if render:
                    appEnv.show_map()

    mean_time=np.mean(total_times)
    std_time=np.std(total_times)

    print("mean time:", mean_time)
    print("std time:", std_time)

    computation_time = time.time()-tic

    return mean_time, std_time, computation_time

def evaluate_mcmpc_policy(appEnvs, repeat, pickers_x, pickers_speed, sample_num, plan_depth, render=False, verbose=False):

    tic = time.time()

    total_times = []

    for appEnv in appEnvs:
        ori_states = appEnv.save_states()

        for i in range(repeat):
            appEnv.load_states(ori_states)

            # init exp
            t = 0

            while True:
                pickers_x = appEnv.policy_mc_mpc(sample_num=sample_num, plan_depth=plan_depth, verbose=False)
                reach_end, _ = appEnv.step(pickers_x)
                t += 1
                if reach_end:
                    total_times.append(t * appEnv.time_step)
                    if verbose:
                        print("Experiment: ", i)
                        print("reach_end")
                        print("total steps:", t)
                        print("total time:", t * appEnv.time_step, " s")
                    break
                if render:
                    appEnv.show_map()

    mean_time=np.mean(total_times)
    std_time=np.std(total_times)

    computation_time = time.time()-tic

    print("--------------------------------")
    print("Evaluated mcmpc_policy ", repeat, " repeat", len(appEnvs), " envs")

    print("sample_num: ", sample_num)
    print("plan_depth: ", plan_depth)

    print("mean time:", mean_time)
    print("std time:", std_time)
    print("computation_time: ", computation_time)

    return mean_time, std_time, computation_time

def main():

    pickers_x = [1, 3]
    pickers_speed = [2.2, 1.1]
    repeat = 1
    exp_num = 1

    appEnvs = []

    for i in range(exp_num):
        appEnv = appleDenseMap(5, pickers_x, pickers_speed, picker_distance=6, map_width=24)
        appEnvs.append(appEnv)


    evaluate_random_policy(copy.deepcopy(appEnvs), repeat, pickers_x, pickers_speed)

    sample_num_exp = 1
    plan_depth = 1
    mean_times = []
    sample_nums = []

    # for sample_num in range(1, 1+sample_num_exp):
    #         plan_depth = 5

    #         mean_time, _, _ = evaluate_mcmpc_policy(copy.deepcopy(appEnvs), repeat, pickers_x, pickers_speed, sample_num=sample_num, plan_depth=plan_depth, render=False)
    #         mean_times.append(mean_time)
    #         sample_nums.append(sample_num)

    evaluate_mcts_policy(copy.deepcopy(appEnvs), repeat, pickers_x, pickers_speed, sample_num=10, plan_depth=2, render=True)

    # print("sample_nums: ", sample_nums)
    # print("mean_time: ", mean_time)
    # plt.plot(sample_nums, mean_time)

    # pool = Pool() # multi processing
    
    # sample_num_exp = 10
    # depth_num_exp = 10

    # mean_times = np.zeros((sample_num_exp, depth_num_exp))
    # std_times = np.zeros((sample_num_exp, depth_num_exp))
    # computation_times = np.zeros((sample_num_exp, depth_num_exp))
    # args=[]

    # for sample_num in range(1, 1+sample_num_exp):
    #     for plan_depth in range(1, 1+depth_num_exp):
    #         args.append([copy.deepcopy(appEnvs), repeat, pickers_x, pickers_speed, sample_num, plan_depth])

    # results = pool.map(eval_multi_helper, args)


    # idx = 0
    # for sample_num in range(1, 1+sample_num_exp):
    #     for plan_depth in range(1, 1+depth_num_exp):
    #         mean_time, std_time, computation_time = results[idx]
    #         mean_times[sample_num_exp-1,plan_depth-1] = mean_time
    #         std_times[sample_num_exp-1,plan_depth-1] = std_time
    #         computation_times[sample_num_exp-1,plan_depth-1] = computation_time            
    #         idx += 1


    # mean_time, std_time, computation_time = evaluate_mcmpc_policy(copy.deepcopy(appEnvs), repeat, pickers_x, pickers_speed, sample_num=sample_num, plan_depth=plan_depth, render=False)

    # print("sample_nums: ", sample_nums)
    # print("mean_time: ", mean_time)
    # print("computation_time: ", computation_time)

    # plt.plot(sample_nums, mean_time)


if __name__ == "__main__":
    main()
