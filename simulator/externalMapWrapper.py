import numpy as np
from simulator.appleSimulator import appleSimulator
import copy
import time

class ExternalMapWrapper(object):
    """docstring for Environment"""
    def __init__(self, time_step=5, pickers_y_init=[1, 4], pickers_speed_init=[0.6, 0.6], platform_speed_init=0.02, 
                 rear_to_cam_center_meters = 4.22, 
                 cam_view_width_meters = 0.9, 
                 picker_distance_meters = 1.8, 
                 picking_windows_size = 0.9,
                 external_map=None,
                 block_size=0.3):


        # simulator environment related
        self.pickers_y = copy.copy(pickers_y_init)
        self.pickers_speed = copy.copy(pickers_speed_init)
        self.platform_speed = copy.copy(platform_speed_init)

        # physical parameters
        self.rear_to_cam_center_meters = rear_to_cam_center_meters  # meter
        self.cam_view_width_meters = cam_view_width_meters  # meter
        self.picker_distance_meters = picker_distance_meters  # meter
        self.picking_windows_size = picking_windows_size # meter, center at picker x
        self.block_size = block_size

        self.picker_distance_pixels, self.sensing_offset, self.cam_view_width, self.max_sensing_limit_pixel, \
        self.max_picking_limit_pixel, self.planning_map_width = \
        self.digitize_physical_parameters(self.picker_distance_meters, self.cam_view_width_meters, \
                                          self.rear_to_cam_center_meters, self.picking_windows_size, self.block_size)

        self.appEnv = appleSimulator(self.pickers_y, self.pickers_speed, self.platform_speed, time_step,
                                     rear_to_cam_center_meters = self.rear_to_cam_center_meters, 
                                     cam_view_width_meters = self.cam_view_width_meters, 
                                     picker_distance_meters = self.picker_distance_meters, 
                                     picking_windows_size = self.picking_windows_size)

        self.state = self.appEnv.save_states()

        # real environment related
        self.now_platform_x_meter_external = 0.

        if external_map is not None:
            self.valid_experiment_range = external_map.shape[1]
            self.external_map = external_map.copy()
            # print("Use user defined external map")

        else:
            self.valid_experiment_range = 30
            # print("Use random created external map")
            density = 58 # 58 apples /meter
            height = self.appEnv.MAP_HEIGHT
            externalmap_width = self.valid_experiment_range + self.sensing_offset + self.planning_map_width
            self.external_map = self.__generate_random_external_map(height, externalmap_width, self.sensing_offset, density)

        self.__init_internal_map()

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
        planning_map_width = max_sensing_limit_pixel  # Option 2: one way to define internal map is plan up ti the max picking limit (should be faster in calculation)
        

        return picker_distance_pixels, sensing_offset, cam_view_width_pixels, max_sensing_limit_pixel, max_picking_limit_pixel, planning_map_width

    def __init_internal_map(self):
        time_sim, platform_x_meter_sim,  pickers_y_meter_sim, planning_map = self.appEnv.save_states()
        planning_map[:, :self.max_sensing_limit_pixel] = self.external_map[:, :self.max_sensing_limit_pixel]
        planning_map[:, self.max_sensing_limit_pixel:] = 0
        self.state = time_sim, platform_x_meter_sim, pickers_y_meter_sim, planning_map
        self.appEnv.load_states(self.state)

    def count_apple_in_valid_range(self):
        # count = np.sum(self.external_map[:, :(self.valid_experiment_range + self.sensing_offset)])
        count = np.sum(self.external_map[:, :-self.max_sensing_limit_pixel])

        return count

    def max_picking(self):
        picking_time = self.state[0]
        picking_speed_sum = np.sum(self.pickers_speed)
        max_picking = picking_time * picking_speed_sum

        return max_picking

    def __generate_random_external_map(self, height, width, init_offset, density):
        #  generate a fake 'real' fruit map
        dens = 3 # 58 * 0.3 / 6
        external_map = np.random.randint(0, dens, size=[height, width], dtype=np.int32)
        external_map[:, :init_offset] = 0
        return external_map


    def step(self, action_external):
        reach_end = False

        # pick and move forward one step
        state_nxt_sim, num_picked, done = self.appEnv.model(self.state, action_external)
        # print("num_picked", num_picked)
        time_sim, platform_x_meter_sim,  pickers_y_meter_sim, internal_map = state_nxt_sim

        last_platform_x_pixel_external =  int(self.now_platform_x_meter_external/self.block_size)
        self.now_platform_x_meter_external = self.now_platform_x_meter_external + platform_x_meter_sim
        now_platform_x_pixel_external = int(self.now_platform_x_meter_external/self.block_size)



        if now_platform_x_pixel_external + self.max_sensing_limit_pixel > self.external_map.shape[1]:
            # print("Reach end of external map")
            reach_end = True

        else:

            # Introduce new part of external map to internal map
            # simulated range
            self.external_map[:, last_platform_x_pixel_external:last_platform_x_pixel_external + self.sensing_offset] = internal_map[:, :self.sensing_offset]
            
            # real range
            internal_map[:, :self.max_sensing_limit_pixel] = self.external_map[:, now_platform_x_pixel_external : now_platform_x_pixel_external + self.max_sensing_limit_pixel]

            # pilatform x
            platform_x_meter_sim = 0 # always keep the platform in zeros position in the simulation for each external action
            # platform_x_meter_sim = self.now_platform_x_meter_external%self.appEnv.BLOCK_SIZE # always keep the platform in zeros position in the simulation for each external action (same pixel not meters)
            
            self.state = time_sim, platform_x_meter_sim, pickers_y_meter_sim, internal_map
            self.appEnv.load_states(self.state)

        return reach_end



