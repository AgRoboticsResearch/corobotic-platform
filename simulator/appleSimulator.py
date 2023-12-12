from __future__ import print_function

import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import seaborn
import os
from simulator import cutils

class appleSimulator(object):

    """docstring for appleSimulator"""

    def __init__(self, pickers_y_pixel_init, pickers_speed, platform_speed, time_step, 
                 rear_to_cam_center_meters = 4.22, 
                 cam_view_width_meters = 0.9, 
                 picker_distance_meters = 1.8, 
                 picking_windows_size = 0.9,
                 lift_lifting_speed = 0.037,
                 lift_descending_speed = 0.037*2
                 ):

        # env info
        self.TIME_STEP = time_step # in seconds
        self.platform_x_pixel = 0 # same as the last picker's picking limit (in pixel)
        self.platform_x_meter = 0 # (in meter)
        self.frame = 0
        self.BLOCK_SIZE = 0.3 # 0.3 m * 0.3 m square


        # physical parameters
        self.rear_to_cam_center_meters = rear_to_cam_center_meters  # meter
        self.cam_view_width_meters = cam_view_width_meters  # meter
        self.picker_distance_meters = picker_distance_meters  # meter
        self.picking_windows_size = picking_windows_size # meter, center at picker x

        picker_distance_pixels, self.SENING_OFFSET, self.CAM_VEIW_WIDTH, self.max_sensing_limit_pixel, \
        self.max_picking_limit_pixel, planning_map_width = \
        self.digitize_physical_parameters(self.picker_distance_meters,  self.cam_view_width_meters, \
                                          self.rear_to_cam_center_meters, self.picking_windows_size, self.BLOCK_SIZE)

        # density map info
        self.MAP_HEIGHT = 6
        self.MAP_WIDTH = planning_map_width  # (in pixel)


        # platform info
        self.platform_speed = platform_speed # in meter/seconds
        self.LIFTING_SPEED = lift_lifting_speed # m/s
        self.DESCENDING_SPEED = lift_descending_speed # m/s

        # pickers info
        self.time = 0.
        self.pickers_y_pixel = pickers_y_pixel_init
        self.pickers_y_meter = np.asarray(self.pickers_y_pixel) * self.BLOCK_SIZE
        self.pickers_speed = pickers_speed 

        self.PICKER_NUM = len(self.pickers_y_pixel)
        self.PICKER_DISTANCE = picker_distance_pixels # distance between each picker in pixel
        self.PICKING_WINDOWS_SIZE = 1 # half of the (picking window - 1 ) 
        self.Y_LIMIT = [self.PICKING_WINDOWS_SIZE, self.MAP_HEIGHT-self.PICKING_WINDOWS_SIZE-1]

        self.ACTION_1D_DIM = 3 # up, keep, down
        self.ACTION_DIM = self.PICKER_NUM

        self.__generate_internal_map()

        # print("Env created")
        # print("MAP_WIDTH", self.MAP_WIDTH)
        # print("ACTION_DIM", self.ACTION_DIM)

    def digitize_physical_parameters(self, picker_distance_meters, cam_view_width_meters, rear_to_cam_center_meters, picking_windows_size, block_size):
        picker_distance_pixels = int(picker_distance_meters/block_size)
        sensing_offset = int((rear_to_cam_center_meters-cam_view_width_meters/2)/block_size)
        cam_view_width_pixels = int(cam_view_width_meters/block_size)
        picking_windows_size_half_pixels = int(picking_windows_size/block_size) - 1 
        max_sensing_limit_meters = rear_to_cam_center_meters + 0.5 * cam_view_width_meters
        max_picking_limit_meters = picking_windows_size + picker_distance_meters 
        max_sensing_limit_pixel =  int(max_sensing_limit_meters/block_size) + 1
        max_picking_limit_pixel = int(max_picking_limit_meters/block_size)

        # planning_map_width = max_sensing_limit_pixel + max_picking_limit_pixel # Option 1: one way to define internal map is let the picing range fully go over current sensing range (should be better in policy)
        # planning_map_width = max_sensing_limit_pixel  # Option 2: one way to define internal map is plan up ti the max picking limit (should be faster in calculation)
        planning_map_width = 2*max_sensing_limit_pixel # Option 3

        return picker_distance_pixels, sensing_offset, cam_view_width_pixels, max_sensing_limit_pixel, max_picking_limit_pixel, planning_map_width


    def count_apple(self):
        # count apple in the valid area (exclude ending area)
        # num_apples = np.sum(self.map[:, 0:self.MAP_WIDTH - 2*self.PICKING_WINDOWS_SIZE - (self.PICKER_NUM-1) * self.PICKER_DISTANCE - 1])
        # num_apples = np.sum(self.map)
        num_apples = np.sum(self.map[:, 0:15])

        return num_apples

    def theoretic_max_pick(self):
        # theoretically how many apple can be picked in the valid area based on picker's picking speed
        theoretic_time = (self.MAP_WIDTH - 2*self.PICKING_WINDOWS_SIZE - (self.PICKER_NUM-1) * self.PICKER_DISTANCE - 1 - ((self.PICKER_NUM-1) * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE + 1 ))*self.BLOCK_SIZE/self.platform_speed
        theoretic_max_picking_rate = np.sum(self.pickers_speed)
        theoretic_max_pick  = theoretic_time*theoretic_max_picking_rate 
        # print("theoretic_time", theoretic_time)
        # print("theoretic_block", self.MAP_WIDTH - 2*self.PICKING_WINDOWS_SIZE - (self.PICKER_NUM-1) * self.PICKER_DISTANCE - 1 - ((self.PICKER_NUM-1) * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE + 1 ))

        return theoretic_max_pick


    def __generate_internal_map(self):
        # initialize the map keeps internally
        self.map = np.zeros([self.MAP_HEIGHT, self.MAP_WIDTH], dtype=np.int32)

    # def __update_internal_map(self):
    #     # detect apples in external map and put them into front of internal map
    #     cam_left_edge = self.platform_x_pixel + self.SENING_OFFSET
    #     cam_right_edge = cam_left_edge + self.CAM_VEIW_WIDTH 

    #     # self.map[:, self.SENING_OFFSET : self.SENING_OFFSET+self.CAM_VEIW_WIDTH] = self.externalmap[:, cam_left_edge: cam_right_edge]
    #     self.map[:, :] = self.externalmap[:, self.platform_x_pixel: self.platform_x_pixel + self.MAP_WIDTH]
    #     # print('map', self.map.shape)

    def __check_reach_end_andmove(self):
        # move the platform when last column has no apple
        reach_end = False

        farthest_picking_limit = self.platform_x_pixel + (self.PICKER_NUM-1) * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE + 1

        if farthest_picking_limit >= self.MAP_WIDTH:
            reach_end =True
        else:
            if cutils.check_empty_vector(self.map[:, self.platform_x_pixel]):
                self.platform_x_pixel += 1

        return reach_end

    def __check_reach_end(self):
        # move the platform when last column has no apple
        reach_end = False

        # farthest_picking_limit = self.platform_x_pixel + (self.PICKER_NUM-1) * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE + 1
        farthest_picking_limit = self.platform_x_pixel + self.max_sensing_limit_pixel

        if farthest_picking_limit > self.MAP_WIDTH:
            reach_end =True

        return reach_end

    def save_states(self):

        saved_states =[self.time, self.platform_x_meter,  copy.copy(self.pickers_y_meter), copy.copy(self.map)]
        
        return saved_states

    def load_states(self, saved_states):

        # self.time, self.platform_x_meter, self.pickers_y_pixel, self.map = copy.deepcopy(saved_states)

        self.time = saved_states[0]
        self.platform_x_meter = saved_states[1]
        self.pickers_y_meter = copy.copy(saved_states[2])
        self.map = copy.copy(saved_states[3])

        self.pickers_y_pixel = [int(self.pickers_y_meter[i]/self.BLOCK_SIZE) for i in range(len(self.pickers_y_meter))]
        # self.pickers_y_pixel = (self.pickers_y_meter/self.BLOCK_SIZE).astype(np.int)
        self.platform_x_pixel = int(self.platform_x_meter/self.BLOCK_SIZE)


    def step(self, action_t, random_seed):
        # propogate system one time step forward
        self.time += self.TIME_STEP
        self.__move_platform()

        # self.__update_internal_map()

        reach_end = self.__check_reach_end()

        if reach_end:
            return self.save_states(), 0, reach_end
        # print(self.map)
        pickers_speed_temp = self.__move_picker(action_t, self.pickers_speed, self.pickers_y_pixel, self.pickers_y_meter, 
                                                self.Y_LIMIT, self.LIFTING_SPEED, self.DESCENDING_SPEED, self.TIME_STEP, self.BLOCK_SIZE)



        picked_num_sum = cutils.pick_opt(self.map, self.pickers_y_pixel, pickers_speed_temp, self.TIME_STEP, self.platform_x_pixel,
                                     self.PICKER_DISTANCE, self.PICKING_WINDOWS_SIZE, self.MAP_WIDTH, self.MAP_HEIGHT, random_seed)
        # print(self.map)
        state_nxt = self.save_states()


        return state_nxt, picked_num_sum, reach_end

    def __move_picker(self, action_t, pickers_speed, pickers_y_pixel, pickers_y_meter, y_limit, lifting_speed, descending_speed, time_step, block_size):
            pickers_speed_temp = cutils.move_picker(action_t, pickers_speed, pickers_y_pixel, pickers_y_meter, 
                                                    y_limit, lifting_speed, descending_speed, time_step, block_size)

            return pickers_speed_temp

    def __move_platform(self):
            self.platform_x_meter += self.TIME_STEP * self.platform_speed
            self.platform_x_pixel = int(self.platform_x_meter/self.BLOCK_SIZE)


    def show_map(self, show_picking_window=True, show_counting_window=False, pause=0.0001, save=None):
        plt.ion()

        # plot map
        img = self.map

        if show_picking_window:
            # plot picker and picking range
            picker_img = np.zeros_like(img)

            for picker_i, picker_y in enumerate(self.pickers_y_pixel):
                # picking window boundary check (unnecessary since we have set y limit, just in case)
                picker_x = min(self.platform_x_pixel + picker_i * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE, self.MAP_WIDTH-1)
                # picker_x = picker_i * self.PICKER_DISTANCE + self.PICKING_WINDOWS_SIZE

                pic_win_y_low = max(picker_x - self.PICKING_WINDOWS_SIZE, 0) 
                pic_win_y_high = min(picker_x + self.PICKING_WINDOWS_SIZE, self.MAP_WIDTH)
                
                pic_win_x_low = max(picker_y - self.PICKING_WINDOWS_SIZE, 0) 
                pic_win_x_high = min(picker_y + self.PICKING_WINDOWS_SIZE, self.MAP_HEIGHT)

                picker_img[pic_win_x_low:pic_win_x_high+1, pic_win_y_low:pic_win_y_high+1] = 5
                picker_img[picker_y, picker_x] = 10

            # merge plots
            transparence = 0.5
            img = (1.-transparence)*img + transparence*picker_img

        if show_counting_window:
            counting_img = np.zeros_like(img)
            counting_img[:, 0:self.MAP_WIDTH - 2*self.PICKING_WINDOWS_SIZE - (self.PICKER_NUM-1) * self.PICKER_DISTANCE - 1] = 5
            # merge plots
            transparence = 0.5
            img = (1.-transparence)*img + transparence*counting_img

        plt.clf()
        plt.imshow(img, interpolation='none')

        if save is not None:
            if not os.path.exists(save):
                os.makedirs(save)  
            png_name = save + 'picking%.5i.png' %(self.frame)
            plt.savefig(png_name)
        plt.pause(pause)
        self.frame += 1

    def model(self, state_t, action_t, random_seed=0):
        self.load_states(state_t)
        state_nxt, num_picked, reach_end  = self.step(action_t, random_seed)


        return state_nxt, num_picked, reach_end


def main():
    # Example random policy

    pickers_y_pixel = [1, 3]
    pickers_speed = [2.2, 1.1]
    picker_distance = 6
    appEnv = appleSimulator(5, pickers_y_pixel, pickers_speed, picker_distance=picker_distance, MAP_WIDTH=24)

    for i in range(100):

        # random policy
        pickers_y_pixel = appEnv.policy_random()
        # print("pickers_y_pixel: ", pickers_y_pixel)
        reach_end, _ = appEnv.step(pickers_y_pixel)
        if reach_end:
            print("-------------- results -------------")
            print("Reach_end")
            print("total steps:", i)
            print("total time:", i * appEnv.time_step, " s")
            break
        appEnv.show_map()

if __name__ == "__main__":
    main()
