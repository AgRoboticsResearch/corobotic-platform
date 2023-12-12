# import numpy as np
from c_utils cimport c_pick_random_n, c_pick_random_n_opt, c_check_empty, c_move_picker
from libcpp cimport bool
from libc.stdlib cimport rand
import random

cimport numpy as np

def pick_random_n(np.ndarray[int, ndim=2, mode="c"] area not None, int n): 
    cdef int area_w, area_h, n_picked
    cdef bool empty

    area_w, area_h = area.shape[0], area.shape[1]

    empty = c_pick_random_n(&area[0,0], area_w, area_h, n, &n_picked)

    return empty, n_picked

def check_empty_large(np.ndarray[int, ndim=2, mode="c"] fruit_map not None, int pic_win_y_low, int pic_win_y_high, int pic_win_x_low, int pic_win_x_high):
    cdef bool empty = True
    cdef int x, y

    for x in range(pic_win_x_low, pic_win_x_high):
        for y in range(pic_win_y_low, pic_win_y_high):
            if fruit_map[x, y] != 0:
                empty = False;
                break;
    return empty;

def pick_random_n_opt(np.ndarray[int, ndim=2, mode="c"] fruit_map not None, int n, 
                      int pic_win_y_low, int pic_win_y_high, int pic_win_x_low, int pic_win_x_high): 
    cdef int area_x, area_y, n_picked
    cdef bool empty
    cdef int rand_x, rand_y, i

    area_x = pic_win_x_high - pic_win_x_low
    area_y = pic_win_y_high - pic_win_y_low

    i = 0

    while True:
        if i >= n:
            break
        
        if check_empty_large(fruit_map, pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high):
            empty = True
            break
        
        rand_x = rand() % area_x + pic_win_x_low        
        rand_y = rand() % area_y + pic_win_y_low        
        
        if(fruit_map[rand_x, rand_y] > 0):
            fruit_map[rand_x, rand_y] -= 1
            i += 1

    
    n_picked = i

    return empty, n_picked

def pick(map, pickers_x, pickers_speed, time_step, platform_x, PICKER_DISTANCE, PICKING_WINDOWS_SIZE, MAP_WIDTH, MAP_HEIGHT):

    # propogate system one time step forward
    cdef int picked_num_sum = 0 # apples picked by all pickers in this step
    cdef int picker_y, pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high, picked_num


    for picker_i, picker_x in enumerate(pickers_x):
        # picking window boundary check (unnecessary since we have set y limit, just in case)
        picker_y = min(platform_x + picker_i * PICKER_DISTANCE + PICKING_WINDOWS_SIZE, MAP_WIDTH-1)
        
        pic_win_y_low = max(picker_y - PICKING_WINDOWS_SIZE, 0) 
        pic_win_y_high = min(picker_y + PICKING_WINDOWS_SIZE, MAP_WIDTH)
        
        pic_win_x_low = max(picker_x - PICKING_WINDOWS_SIZE, 0) 
        pic_win_x_high = min(picker_x + PICKING_WINDOWS_SIZE, MAP_HEIGHT)

        area = map[pic_win_x_low:pic_win_x_high+1, pic_win_y_low:pic_win_y_high+1]

        area = area.copy(order='C')

        _, picked_num = pick_random_n(area, n=time_step*pickers_speed[picker_i])

        map[pic_win_x_low:pic_win_x_high+1, pic_win_y_low:pic_win_y_high+1] = area
        picked_num_sum += picked_num

    return picked_num_sum

def pick_opt(np.ndarray[int, ndim=2, mode="c"] fruit_map not None, pickers_x, pickers_speed, time_step, platform_x, PICKER_DISTANCE, PICKING_WINDOWS_SIZE, MAP_WIDTH, MAP_HEIGHT, seed):

    # propogate system one time step forward
    cdef int picked_num_sum = 0 # apples picked by all pickers in this step
    cdef int picker_y, pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high, picked_num
    cdef int picker_i, picker_x
    cdef int picker_num = len(pickers_x)
    cdef int n_picked
    cdef int fruit_map_x, fruit_map_y
    cdef int pick_capacity
    cdef float residue


    fruit_map_x, fruit_map_y = fruit_map.shape[0], fruit_map.shape[1]


    for picker_i in range(picker_num):
        # picking window boundary check (unnecessary since we have set y limit, just in case)
        picker_y = min(platform_x + picker_i * PICKER_DISTANCE + PICKING_WINDOWS_SIZE, MAP_WIDTH-1)
        
        pic_win_y_low = max(picker_y - PICKING_WINDOWS_SIZE, 0) 
        pic_win_y_high = min(picker_y + PICKING_WINDOWS_SIZE, MAP_WIDTH)+1
        
        pic_win_x_low = max(pickers_x[picker_i] - PICKING_WINDOWS_SIZE, 0) 
        pic_win_x_high = min(pickers_x[picker_i] + PICKING_WINDOWS_SIZE, MAP_HEIGHT)+1 

        # _, picked_num = pick_random_n_opt(fruit_map, time_step*pickers_speed[picker_i],
        #                              pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high)

        pick_capacity = int(time_step*pickers_speed[picker_i])
        residue = time_step*pickers_speed[picker_i] - pick_capacity

        if random.random() < residue:
            pick_capacity += 1


        c_pick_random_n_opt(&fruit_map[0,0], fruit_map_x, fruit_map_y , pick_capacity, &picked_num,
                                     pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high, seed)

        picked_num_sum += picked_num

    return picked_num_sum

def check_empty_vector(vector):
    cdef int i
    cdef bool empty = True
    for i in vector:
        if i != 0:
            empty = False
            break
    return empty


def move_picker(action_t, pickers_speed, pickers_y_pixel, pickers_y_meter, y_limit, lifting_speed, descending_speed, time_step, block_size):
    cdef picker_num = len(pickers_y_pixel)
    cdef int action_i, i;
    pickers_speed_temp = [1.] * picker_num # if move at this time step, picker's picking speed will be affected

    for i in range(picker_num):
        action_i = action_t[i]
        pickers_speed_temp[i] = pickers_speed[i]

        if action_i == 0: # keep still
            # pickers_speed_temp[i] = pickers_speed[i]
            pass

        elif action_i == 1: # move up
            pickers_y_meter[i] -= time_step *  lifting_speed
            pickers_y_meter[i] = max(y_limit[0]*block_size, pickers_y_meter[i])

            # pickers_y_pixel[i] = max(y_limit[0], pickers_y_pixel[i] - 1)

        elif action_i == 2: # move down
            pickers_y_meter[i] += time_step * descending_speed 
            pickers_y_meter[i] = min(y_limit[1]*block_size, pickers_y_meter[i])

            # pickers_y_pixel[i] = min(y_limit[1], pickers_y_pixel[i] + 1)

        pickers_y_pixel[i] = int(pickers_y_meter[i]/block_size)

    return pickers_speed_temp

def get_action(action_1d_dim):

    cdef double action_t[2];
    action_t[0] = rand()%action_1d_dim
    action_t[1] = rand()%action_1d_dim

    return action_t

# def save_states(double time, double platform_x_meter, 
#     np.ndarray[double, ndim=1, mode="c"] pickers_y_meter not None, np.ndarray[int, ndim=2, mode="c"] fruit_map not None):

#     cdef np.ndarray[double, ndim=1, mode="c"] pickers_y_meter_save = pickers_y_meter
#     cdef np.ndarray[int, ndim=2, mode="c"] fruit_map_save = fruit_map

#     saved_states =[time, platform_x_meter, pickers_y_meter_save, fruit_map_save]

#     return saved_states

