from __future__ import print_function
from mcts.mcts_sparse import MctsSparse
from mcts.mcts import Mcts, MctsSpw, MctsDpw
from simulator.controller import RandomController, DefaultController
from simulator.appleSimulator import appleSimulator
from simulator.externalMapWrapper import ExternalMapWrapper

from mcts.mcts_utils import ModelWrapper

import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import random
from multiprocessing import Pool

def generate_random_external_map(height, width, init_offset, density, seed=0):
    # ---------------
    # params in: 
    # height, width: in pixels
    # init_offset: distance from platform x to edge of camera view in pixels

    np.random.seed(seed)
    external_map = np.random.randint(0, density, size=[height, width], dtype=np.int32)
    external_map[:, :init_offset] = 0
    return external_map

def append_random_to_zero(x, results_mean, results_sdt, random_mean, random_std):
    x =  np.concatenate(([0], x))
    results_mean = np.concatenate(([random_mean],results_mean))
    results_sdt = np.concatenate(([random_std], results_sdt))
    return x, results_mean, results_sdt

def plot_results(x, y, y_std, xlabel="x", ylabel="y", legend='SparseSampling', title="Title", style='ro-'):
    fig = plt.figure().gca()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cis = (y - y_std, y + y_std)
    fig.fill_between(x,cis[0],cis[1],alpha=0.2)
    fig.plot(x, y, style, label=legend)
    # fig.margins(y=0)
    # plt.show()

def plot_3results(x, random_means, random_stds, default_means, default_stds, sparse_means, sparse_stds, xlabel="x", ylabel="y", title="Title",
                 random_label="Random Policy", default_label="Default Policy", sparse_label="Sparse Sampling Pilicy"):
    fig = plt.figure(figsize=[7, 5]).gca()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # random 
    cis = (random_means - random_stds, random_means + random_stds)
    fig.fill_between(x,cis[0],cis[1],alpha=0.2)
    fig.plot(x, random_means, 'ro--', label=random_label)
    
    # default
    cis = (default_means - default_stds, default_means + default_stds)
    fig.fill_between(x,cis[0],cis[1],alpha=0.2)
    fig.plot(x, default_means, 'g*--', label=default_label)

    # sparse
    cis = (sparse_means - sparse_stds, sparse_means + sparse_stds)
    fig.fill_between(x,cis[0],cis[1],alpha=0.2)
    fig.plot(x, sparse_means, 'b^--', label=sparse_label)

    # fig.margins(y=0)
    # plt.show()

def plot_bars(means, stds):

    print("means ", means)
    print("stds ", stds)
    
    fig, ax = plt.subplots()

    n_groups = len(means)
    index = np.arange(n_groups)

    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, means, bar_width,
                    alpha=opacity, color='b',
                    yerr=stds, error_kw=error_config,
                    label='Policy')

    ax.set_xlabel('Policy')
    ax.set_ylabel('Picked Percent %')
#     ax.set_title('Title')
    ax.set_xticks(index)
    ax.set_xticklabels(('Random', 'Default', 'Sparse Sampling', 'D', 'E'))
#     ax.legend()

    fig.tight_layout()
    plt.show()


def plot_baseline(x, y, legend='baseline', style='--'):
    ys = []
    for i in range(len(x)):
        ys.append(y)

    plt.plot(x, ys, style, label=legend )

def mcts_sparse_experiment(time_step, repeat, external_map, plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample,
                           pickers_y_init=[1, 4], pickers_speed_init=[0.6, 0.6], platform_speed_init=0.02, gamma=0.99,
                           rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                           render=False, verbose=False, save=False):
    # repeat:  number
    # other inputs are lists   
    args = []
    print("repeat: ", repeat)

    for i in range(repeat):
        for d in plan_depth:
            for n in sample_num_per_action:
                for m in extra_search_depth:
                    for k in extra_search_sample:
                        seed = i
                        ext_map = copy.copy(external_map)

                        args.append([time_step, ext_map, d, n, m, k, seed,
                                     pickers_y_init, pickers_speed_init, platform_speed_init, gamma,
                                     rear_to_cam_center_meters, cam_view_width_meters, picker_distance_meters, picking_windows_size, 
                                     render, verbose, save])

    pool = Pool() # multi processing
    results = pool.map(eval_multi_helper_mcts_sparse, args)
    pool.close()
    pool.join()
    
    results = np.asarray(results)
    total_steps = results[:, 2]        

    picked_percent = results[:,0]
    picked_percent = picked_percent.reshape((repeat, -1))
    picked_percent_mean = np.mean(picked_percent, axis=0)
    picked_percent_std = np.std(picked_percent, axis=0)

    picked_number = results[:,1]
    picked_number = picked_number.reshape((repeat, -1))
    picked_number_mean = np.mean(picked_number, axis=0)
    picked_number_std = np.std(picked_number, axis=0)

    print("picked_percent ", picked_percent)
    print("picked_percent_mean ", picked_percent_mean)
    print("picked_percent_std ", picked_percent_std)

    return picked_percent, picked_percent_mean, picked_percent_std, picked_number, picked_number_mean, picked_number_std, total_steps

def eval_multi_helper_mcts_sparse(args):
    # auxiliary funciton to make evaluating multiprocessing with multi arguments
    return evaluate_mcts_sparse(*args)

def policy_experiment(time_step, repeat, external_map, policy_fn, pickers_y_init=[1, 4], pickers_speed_init=[0.6, 0.6], platform_speed_init=0.02, 
                             rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                             verbose=False):

    
    results = []
    for i in range(repeat):
        ext_map = copy.copy(external_map)
        result = evaluate_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, policy_fn, 
                            rear_to_cam_center_meters=rear_to_cam_center_meters,
                            cam_view_width_meters=cam_view_width_meters,
                            picker_distance_meters=picker_distance_meters,
                            picking_windows_size=picking_windows_size,
                            render=False, verbose=True, save=None)
        results.append(result)
        
    results = np.asarray(results)
    total_steps = results[:, 2]        

    picked_percent = results[:,0]
    picked_percent = picked_percent.reshape((repeat, -1))
    picked_percent_mean = np.mean(picked_percent, axis=0)
    picked_percent_std = np.std(picked_percent, axis=0)

    picked_number = results[:,1]
    picked_number = picked_number.reshape((repeat, -1))
    picked_number_mean = np.mean(picked_number, axis=0)
    picked_number_std = np.std(picked_number, axis=0)

    if verbose:
        print("picked_percent ", picked_percent)
        print("picked_percent_mean ", picked_percent_mean)
        print("picked_percent_std ", picked_percent_std)

    return picked_percent, picked_percent_mean, picked_percent_std, picked_number, picked_number_mean, picked_number_std, total_steps

def random_policy_experiment(time_step, repeat, external_map, pickers_y_init=[1, 4], pickers_speed_init=[0.6, 0.6], platform_speed_init=0.02, 
                             rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                             verbose=False):

    
    results = []
    for i in range(repeat):
        ext_map = copy.copy(external_map)
        result = evaluate_random_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
                            rear_to_cam_center_meters=rear_to_cam_center_meters,
                            cam_view_width_meters=cam_view_width_meters,
                            picker_distance_meters=picker_distance_meters,
                            picking_windows_size=picking_windows_size,
                            render=False, verbose=True, save=None)
        results.append(result)
        
    results = np.asarray(results)
    total_steps = results[:, 2]        

    picked_percent = results[:,0]
    picked_percent = picked_percent.reshape((repeat, -1))
    picked_percent_mean = np.mean(picked_percent, axis=0)
    picked_percent_std = np.std(picked_percent, axis=0)

    picked_number = results[:,1]
    picked_number = picked_number.reshape((repeat, -1))
    picked_number_mean = np.mean(picked_number, axis=0)
    picked_number_std = np.std(picked_number, axis=0)

    if verbose:
        print("picked_percent ", picked_percent)
        print("picked_percent_mean ", picked_percent_mean)
        print("picked_percent_std ", picked_percent_std)

    return picked_percent, picked_percent_mean, picked_percent_std, picked_number, picked_number_mean, picked_number_std, total_steps

def default_policy_experiment(time_step, repeat, external_map, pickers_y_init=[1, 4], pickers_speed_init=[0.6, 0.6], platform_speed_init=0.02, 
                             rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                             verbose=False):

    
    results = []
    for i in range(repeat):
        ext_map = copy.copy(external_map)
        result = evaluate_default_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
                            rear_to_cam_center_meters=rear_to_cam_center_meters,
                            cam_view_width_meters=cam_view_width_meters,
                            picker_distance_meters=picker_distance_meters,
                            picking_windows_size=picking_windows_size,
                            render=False, verbose=True, save=None)

        results.append(result)

    results = np.asarray(results)
    total_steps = results[:, 2]        

    picked_percent = results[:,0]
    picked_percent = picked_percent.reshape((repeat, -1))
    picked_percent_mean = np.mean(picked_percent, axis=0)
    picked_percent_std = np.std(picked_percent, axis=0)

    picked_number = results[:,1]
    picked_number = picked_number.reshape((repeat, -1))
    picked_number_mean = np.mean(picked_number, axis=0)
    picked_number_std = np.std(picked_number, axis=0)

    if verbose:
        print("picked_percent ", picked_percent)
        print("picked_percent_mean ", picked_percent_mean)
        print("picked_percent_std ", picked_percent_std)

    return picked_percent, picked_percent_mean, picked_percent_std, picked_number, picked_number_mean, picked_number_std, total_steps

def evaluate_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, policy_fn,
                            rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                            render=False, verbose=False, save=None ):

    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)
    policy_fn = policy_fn(env.appEnv)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()
    total_steps = 0

    while True:
        if render:
            env.appEnv.show_map(save=save)

        action = policy_fn.get_action(env.state)
        done = env.step(action)
        total_steps += 1

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Total Step: ", total_steps)


    return picked_percent, picked_number, total_steps

def evaluate_default_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
                            rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                            render=False, verbose=False, save=None ):

    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()
    total_steps = 0

    while True:
        if render:
            env.appEnv.show_map(save=save)

        action = [0, 0]
        done = env.step(action)
        total_steps += 1

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("Default policy")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Total Step: ", total_steps)


    return picked_percent, picked_number, total_steps

def evaluate_random_policy(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
                          rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                          render=False, verbose=False, save=None ):


    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)
    random_policy_fn = RandomController(env.appEnv)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()
    total_steps = 0

    while True:
        if render:
            env.appEnv.show_map(save=save)

        action = random_policy_fn.get_action(env.state)
        done = env.step(action)
        total_steps += 1

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("Random policy")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Total Step: ", total_steps)


    return picked_percent, picked_number, total_steps

def evaluate_mcts_sparse(time_step, external_map, plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, random_seed, 
                         pickers_y_init, pickers_speed_init, platform_speed_init, gamma=0.99,
                         rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                         render=False, verbose=False, save=None):
    if verbose: 
        print("--------------------------------")
        print("Evaluating mcts sparse ......")
        print("sample num per action (C): ", sample_num_per_action)
        print("plan_depth (H) : ", plan_depth)
        print("extra_search_depth (M): ", extra_search_depth)
        print("extra_search_sample (K): ", extra_search_sample)
        print("gamma: ", gamma)

    # set random seeds (used for the reason that multi processing use same random seed all the time) 
    np.random.seed(random_seed) 
    random.seed(random_seed)

    # environment
    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)
    # plt.figure()
    # plt.imshow(env.external_map)

    random_policy_fn = RandomController(env.appEnv)
    model_fn = ModelWrapper(env.appEnv)
    sparse_sampling = MctsSparse(env.appEnv, plan_depth, gamma, model_fn, random_policy_fn, 
                                extra_search_depth=extra_search_depth, extra_search_sample=extra_search_sample)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()

    step = 0

    while True:
        # t_step_start = time.time()

        if render:
            env.appEnv.show_map(save=save)
        action, best_q = sparse_sampling.run(sample_num_per_action=sample_num_per_action)
        # print("action: ", action)
        # print("Step Time: ", time.time() - t_step_start)

        done = env.step(action)
        step += 1
        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    max_picking = env.max_picking()
    effective_picking_efficiency = (num_apple_start - num_apple_left)/max_picking

    if verbose:
        print("--------------------  Results  -------------------- ")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked apple:", num_apple_start - num_apple_left)
        print("Picking time (s): ", env.state[0])

        print("Time", cost_time)
        print("Computation time per step %.2f s" %(cost_time/step))


        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Effective Picking Efficiency: %.2f %%" %(effective_picking_efficiency*100))
        print("Total Step: ", step)
    # plt.figure()
    # plt.imshow(env.external_map)

    return picked_percent, picked_number, step


# Adapative speed evaluation

def evaluate_speed(platform_speed, policy_fn, env, pickers_y_init, pickers_speed_init,
                   time_step=5, rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                   verbose=False, render=False):
    
    # speed optimizer simulator
    pickers_speed = env.appEnv.pickers_speed
    pickers_y_init = env.appEnv.pickers_y_pixel
    speed_optimizer_env= ExternalMapWrapper(time_step=time_step, 
                                             pickers_y_init=pickers_y_init, 
                                             pickers_speed_init=pickers_speed_init,  
                                             platform_speed_init=platform_speed,
                                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                             cam_view_width_meters=cam_view_width_meters, 
                                             picker_distance_meters=picker_distance_meters, 
                                             picking_windows_size=picking_windows_size,
                                             external_map=env.appEnv.map)

    policy_fn = policy_fn(speed_optimizer_env.appEnv)
    num_apple_start = speed_optimizer_env.count_apple_in_valid_range()

    t_start = time.time()

    while True:
        if render:
            speed_optimizer_env.appEnv.show_map()

        action = policy_fn.get_action(env.state)
        done = speed_optimizer_env.step(action)

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = speed_optimizer_env.count_apple_in_valid_range()
    if num_apple_start!=0:
        picked_percent = (1 - num_apple_left/num_apple_start)
    else:
        picked_percent = 1.
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Evaluating time %.2fs" % cost_time)


    return picked_percent


def evaluate_speed_mcts(platform_speed, env, pickers_y_init, pickers_speed_init,
                        plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, gamma=0.99,
                        time_step=5, rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                        verbose=False, render=False):
    
    # speed optimizer simulator
    pickers_speed = env.appEnv.pickers_speed
    pickers_y_init = env.appEnv.pickers_y_pixel
    speed_optimizer_env= ExternalMapWrapper(time_step=time_step, 
                                             pickers_y_init=pickers_y_init, 
                                             pickers_speed_init=pickers_speed_init,  
                                             platform_speed_init=platform_speed,
                                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                             cam_view_width_meters=cam_view_width_meters, 
                                             picker_distance_meters=picker_distance_meters, 
                                             picking_windows_size=picking_windows_size,
                                             external_map=env.appEnv.map)
    
    random_policy_fn = RandomController(speed_optimizer_env.appEnv)
    model_fn = ModelWrapper(speed_optimizer_env.appEnv)

    sparse_sampling = MctsSparse(speed_optimizer_env.appEnv, plan_depth, gamma, model_fn, random_policy_fn, 
                                 extra_search_depth=extra_search_depth, extra_search_sample=extra_search_sample)


    num_apple_start = speed_optimizer_env.count_apple_in_valid_range()

    t_start = time.time()

    while True:
        if render:
            speed_optimizer_env.appEnv.show_map()

        action, best_q = sparse_sampling.run(sample_num_per_action=sample_num_per_action)
        done = speed_optimizer_env.step(action)

        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = speed_optimizer_env.count_apple_in_valid_range()
    if num_apple_start!=0:
        picked_percent = (1 - num_apple_left/num_apple_start)
    else:
        picked_percent = 1
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Evaluating time %.2fs" % cost_time)

    return picked_percent


def evaluate_mcts_sparse_opt_speed(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
                                   plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, gamma=0.99, random_seed=0,
                                  rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                                  platform_speed_options = [0.08, 0.06, 0.04, 0.02, 0.01],
                                  min_picking_percentage = 0.95,
                                  heuristic_policy_fn=None, speed_opt_every_step=False,
                                  render=False, verbose=False, save=None):
    if verbose: 
        print("--------------------------------")
        print("Evaluating mcts sparse ......")
        print("sample num per action (C): ", sample_num_per_action)
        print("plan_depth (H) : ", plan_depth)
        print("extra_search_depth (M): ", extra_search_depth)
        print("extra_search_sample (K): ", extra_search_sample)

    # set random seeds (used for the reason that multi processing use same random seed all the time) 
    np.random.seed(random_seed) 
    random.seed(random_seed)


    # environment
    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)

    random_policy_fn = RandomController(env.appEnv)
    model_fn = ModelWrapper(env.appEnv)
    sparse_sampling = MctsSparse(env.appEnv, plan_depth, gamma, model_fn, random_policy_fn, 
                                extra_search_depth=extra_search_depth, extra_search_sample=extra_search_sample)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()

    total_steps = 0
    speed_planing_valid_count = np.inf
    valid_planning_timestep = 0
    speed_profile = []

    while True:
        speed_planing_valid_count += 1
        t_step_start = time.time()

        if render:
            env.appEnv.show_map(save=save)
        
        # ----------- Speed optimization ----------------
        if speed_planing_valid_count > valid_planning_timestep or speed_opt_every_step:
            if heuristic_policy_fn is None:
                opt_speed = speed_optimizer_mcts(platform_speed_options, env, pickers_y_init, pickers_speed_init,
                                                plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, 
                                                time_step=time_step,
                                                rear_to_cam_center_meters=rear_to_cam_center_meters,
                                                cam_view_width_meters=cam_view_width_meters, 
                                                picker_distance_meters=picker_distance_meters,
                                                picking_windows_size=picking_windows_size, 
                                                min_picking_percentage=min_picking_percentage)
            else:
                opt_speed = speed_optimizer_mcts_heuristic_speed_up(platform_speed_options,heuristic_policy_fn, env, pickers_y_init, pickers_speed_init,
                                                plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, 
                                                time_step=time_step,
                                                rear_to_cam_center_meters=rear_to_cam_center_meters,
                                                cam_view_width_meters=cam_view_width_meters, 
                                                picker_distance_meters=picker_distance_meters,
                                                picking_windows_size=picking_windows_size, 
                                                min_picking_percentage=min_picking_percentage)            

            # Set speed
            env.platform_speed = opt_speed
            env.appEnv.platform_speed = opt_speed

            # Calculate speed valid time
            valid_planning_range = env.appEnv.map.shape[1] - env.max_sensing_limit_pixel - (env.appEnv.PICKER_DISTANCE + 2*env.appEnv.PICKING_WINDOWS_SIZE)
            valid_planning_period = (valid_planning_range*env.block_size)/env.platform_speed 
            valid_planning_timestep = min(5, int((valid_planning_period/time_step)/2)) # divide by 2 to increase speed planning frequency in order to deal with sensing changing        
            speed_planing_valid_count = 0

            if not speed_opt_every_step:
                print("planning speed: ", opt_speed)
                print("planning speed valid for : ", valid_planning_timestep, " step")

        speed_profile.append(env.appEnv.platform_speed)
        # -----------------------------------------------         
        
        action, best_q = sparse_sampling.run(sample_num_per_action=sample_num_per_action)
        done = env.step(action)
        total_steps += 1


        print("Step Time: ", time.time() - t_step_start)
        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    max_picking = env.max_picking()
    effective_picking_efficiency = (num_apple_start - num_apple_left)/max_picking

    if verbose:
        print("--------------------  Results  -------------------- ")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked apple:", num_apple_start - num_apple_left)
        print("Picking time (s): ", env.state[0])

        print("Time", cost_time)
        print("Computation time per step %.2f s" %(cost_time/total_steps))


        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Effective Picking Efficiency: %.2f %%" %(effective_picking_efficiency*100))
        print("Total Steps: ", total_steps)

    return min_picking_percentage, picked_percent, picked_number, total_steps, speed_profile

def evaluate_mcts_sparse_opt_speed_multi_helper(args):
    return evaluate_mcts_sparse_opt_speed(*args)

def evaluate_default_policy_opt_speed_multi_helper(args):
    return evaluate_default_policy_opt_speed(*args)

def evaluate_policy_opt_speed_multi_helper(args):
    return evaluate_policy_opt_speed(*args)

def evaluate_policy_opt_speed(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, policy_fn,
                              rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                              platform_speed_options=[0.08, 0.06, 0.04, 0.02, 0.01],
                              min_picking_percentage=0.95,  speed_opt_every_step=False,
                              render=False, verbose=False, save=None):


    env = ExternalMapWrapper(time_step=time_step, 
                             pickers_y_init=pickers_y_init, 
                             pickers_speed_init=pickers_speed_init,  
                             platform_speed_init=platform_speed_init,
                             rear_to_cam_center_meters=rear_to_cam_center_meters, 
                             cam_view_width_meters=cam_view_width_meters, 
                             picker_distance_meters=picker_distance_meters, 
                             picking_windows_size=picking_windows_size,
                             external_map=external_map)

    num_apple_start = env.count_apple_in_valid_range()

    t_start = time.time()

    total_steps = 0
    speed_profile = []
    policy = policy_fn(env.appEnv)

    speed_planing_valid_count = np.inf
    valid_planning_timestep = 0


    while True:
        if render:
            env.appEnv.show_map(save=save)
            
        # ----------- Speed optimization ----------------
        if (speed_planing_valid_count > valid_planning_timestep) or speed_opt_every_step:

            opt_speed = speed_optimizer(platform_speed_options, policy_fn, env, pickers_y_init, pickers_speed_init,
                                        time_step=time_step,
                                        rear_to_cam_center_meters=rear_to_cam_center_meters,
                                        cam_view_width_meters=cam_view_width_meters, 
                                        picker_distance_meters=picker_distance_meters,
                                        picking_windows_size=picking_windows_size, 
                                        min_picking_percentage=min_picking_percentage)

            # Set speed
            env.platform_speed = opt_speed
            env.appEnv.platform_speed = opt_speed

            # Calculate speed valid time
            valid_planning_range = env.appEnv.map.shape[1] - env.max_sensing_limit_pixel - (env.appEnv.PICKER_DISTANCE + 2*env.appEnv.PICKING_WINDOWS_SIZE)
            valid_planning_period = (valid_planning_range*env.block_size)/env.platform_speed 
            valid_planning_timestep = min(5, int((valid_planning_period/time_step)/2)) # divide by 2 to increase speed planning frequency in order to deal with sensing changing        
            speed_planing_valid_count = 0

            if not speed_opt_every_step:
                print("planning speed: ", opt_speed)
                print("planning speed valid for : ", valid_planning_timestep, " step") 

        speed_profile.append(env.appEnv.platform_speed)

        # -----------------------------------------------         
        action = policy.get_action(env.state)
        done = env.step(action)
        total_steps += 1
        if done:
            break

    cost_time = time.time() - t_start
    num_apple_left = env.count_apple_in_valid_range()
    picked_percent = (1 - num_apple_left/num_apple_start)
    picked_number = (num_apple_start - num_apple_left)

    if verbose:
        print("-----------------")
        print("Apple num: ", num_apple_start)
        print("Left apple: ", num_apple_left)
        print("Picked Percent: %.2f %%" %(picked_percent*100))
        print("Total Steps: ", total_steps)


    return min_picking_percentage, picked_percent, picked_number, total_steps, speed_profile


# def evaluate_default_policy_opt_speed(time_step, external_map, pickers_y_init, pickers_speed_init, platform_speed_init, 
#                                       rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
#                                       platform_speed_options=[0.08, 0.06, 0.04, 0.02, 0.01],
#                                       min_picking_percentage=0.95,
#                                       render=False, verbose=False, save=None):


#     env = ExternalMapWrapper(time_step=time_step, 
#                              pickers_y_init=pickers_y_init, 
#                              pickers_speed_init=pickers_speed_init,  
#                              platform_speed_init=platform_speed_init,
#                              rear_to_cam_center_meters=rear_to_cam_center_meters, 
#                              cam_view_width_meters=cam_view_width_meters, 
#                              picker_distance_meters=picker_distance_meters, 
#                              picking_windows_size=picking_windows_size,
#                              external_map=external_map)

#     num_apple_start = env.count_apple_in_valid_range()

#     t_start = time.time()

#     total_steps = 0
#     opt_speeds_stats = []
#     while True:
#         if render:
#             env.appEnv.show_map(save=save)
            
#         # ----------- Speed optimization ----------------
#         policy_fn = DefaultController
#         opt_speed = speed_optimizer(platform_speed_options, policy_fn, env, pickers_y_init, pickers_speed_init,
#                                     time_step=time_step,
#                                     rear_to_cam_center_meters=rear_to_cam_center_meters,
#                                     cam_view_width_meters=cam_view_width_meters, 
#                                     picker_distance_meters=picker_distance_meters,
#                                     picking_windows_size=picking_windows_size, 
#                                     min_picking_percentage=min_picking_percentage)

#         # Set speed
#         env.platform_speed = opt_speed
#         env.appEnv.platform_speed = opt_speed
#         opt_speeds_stats.append(opt_speed)
#         # -----------------------------------------------         
#         action = [0, 0]
#         done = env.step(action)
#         total_steps += 1
#         if done:
#             break

#     cost_time = time.time() - t_start
#     num_apple_left = env.count_apple_in_valid_range()
#     picked_percent = (1 - num_apple_left/num_apple_start)
#     picked_number = (num_apple_start - num_apple_left)

#     if verbose:
#         print("Default policy")
#         print("Apple num: ", num_apple_start)
#         print("Left apple: ", num_apple_left)
#         print("Picked Percent: %.2f %%" %(picked_percent*100))
#         print("Total Steps: ", total_steps)


#     return min_picking_percentage, picked_percent, picked_number, total_steps


def speed_optimizer(platform_speed_options, policy_fn, env, pickers_y_init, pickers_speed_init,
                    time_step=5, rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                    min_picking_percentage=0.95):
    
    # calculate an optimal speed that can achieve min_picking_percentage under policy_fn

    opt_speed = None

    for i in range(len(platform_speed_options) - 1):
        platform_speed = platform_speed_options[i]

        
        picking_percentage = evaluate_speed(platform_speed, policy_fn, env, pickers_y_init, pickers_speed_init,
                                           time_step=time_step, 
                                           rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                           cam_view_width_meters=cam_view_width_meters, 
                                           picker_distance_meters=picker_distance_meters, 
                                           picking_windows_size=picking_windows_size, 
                                           render=False, verbose=False)

        if picking_percentage >= min_picking_percentage:
            opt_speed = platform_speed
            break
    if opt_speed is None:
        opt_speed = platform_speed_options[-1]

    # print("OPT Speed: ", opt_speed)
    # print("picking_percentage: ", picking_percentage)
    # print("min_picking_percentage: ", min_picking_percentage)


    return opt_speed

def speed_optimizer_mcts(platform_speed_options, env, pickers_y_init, pickers_speed_init,
                        plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, gamma=0.99,
                        time_step=5, rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                        min_picking_percentage=0.95):
    
    # calculate an optimal speed that can achieve min_picking_percentage under mcts
    
    opt_speed = None
    
    for i in range(len(platform_speed_options) - 1):
        platform_speed = platform_speed_options[i]

        picking_percentage = evaluate_speed_mcts(platform_speed, env, pickers_y_init, pickers_speed_init,
                                               plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, 
                                               rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                               cam_view_width_meters=cam_view_width_meters, 
                                               picker_distance_meters=picker_distance_meters, 
                                               picking_windows_size=picking_windows_size, 
                                               render=False, verbose=False)

        if picking_percentage >= min_picking_percentage:
            opt_speed = platform_speed
            break

    if opt_speed is None:
        opt_speed = platform_speed_options[-1]

    return opt_speed



def speed_optimizer_mcts_heuristic_speed_up(platform_speed_options, heuristic_policy_fn, env, pickers_y_init, pickers_speed_init,
                                            plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, gamma=0.99,
                                            time_step=5, rear_to_cam_center_meters=4.22, cam_view_width_meters=0.9, picker_distance_meters=1.8, picking_windows_size=0.9, 
                                            min_picking_percentage=0.95, verbose=True):
    # Calculate an optimal speed that can achieve min_picking_percentage under mcts.
    # To speedup computation, calculate the optimal speed under an heuristic policy first. We assume the mcts is dominate any heuristic policy.
    # So, if the heuristic policy can already achieve min_picking_percentage, the mcts can also achieve it.
    
    opt_speed = None
    if verbose:
        print("------------ speed optmizing.... ----------")
    
    for i in range(len(platform_speed_options) - 1):
        platform_speed = platform_speed_options[i]
        if verbose:
            print("platform_speed: ", platform_speed)

        picking_percentage_heuristic = evaluate_speed(platform_speed, heuristic_policy_fn, env, pickers_y_init, pickers_speed_init,
                                           time_step=time_step, 
                                           rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                           cam_view_width_meters=cam_view_width_meters, 
                                           picker_distance_meters=picker_distance_meters, 
                                           picking_windows_size=picking_windows_size, 
                                           render=False, verbose=False)

        if picking_percentage_heuristic >= min_picking_percentage:
            opt_speed = platform_speed
            if verbose:
                print("achieved using heuristic")
            break
        else:
            picking_percentage = evaluate_speed_mcts(platform_speed, env, pickers_y_init, pickers_speed_init,
                                                   plan_depth, sample_num_per_action, extra_search_depth, extra_search_sample, 
                                                   rear_to_cam_center_meters=rear_to_cam_center_meters, 
                                                   cam_view_width_meters=cam_view_width_meters, 
                                                   picker_distance_meters=picker_distance_meters, 
                                                   picking_windows_size=picking_windows_size, 
                                                   render=False, verbose=False)

            if picking_percentage >= min_picking_percentage:
                opt_speed = platform_speed
                break
        if opt_speed is None:
            opt_speed = platform_speed_options[-1]

    return opt_speed


