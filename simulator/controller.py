import numpy as np
# from simulator import cutils

class RandomController(object):
    """docstring for RandomController"""
    def __init__(self, env):
        self.env = env
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    # def get_action_c(self, s_t):
    #     # return np.random.randint(0, self.ACTION_1D_DIM, size=self.env.ACTION_DIM)
    #     return cutils.get_action(self.ACTION_1D_DIM)

    def get_action(self, s_t):
        return np.random.randint(0, self.ACTION_1D_DIM, size=self.env.ACTION_DIM)

    def get_action_1d(self, s_t):
        return np.random.randint(0, self.ACTION_DIM_FLAT)

class DefaultController(object):
    """docstring for DefaultController"""
    def __init__(self, env):
        self.env = env
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    # def get_action_c(self, s_t):
    #     # return np.random.randint(0, self.ACTION_1D_DIM, size=self.env.ACTION_DIM)
    #     return cutils.get_action(self.ACTION_1D_DIM)

    def get_action(self, s_t):
        return np.zeros([self.ACTION_DIM], dtype=np.int32)

class HeuristicControllerDensity(object):
    def __init__(self, env):
        self.env = env
        self.ACTION_DIM = self.env.ACTION_DIM
        self.ACTION_1D_DIM = self.env.ACTION_1D_DIM
        self.ACTION_DIM_FLAT = self.ACTION_1D_DIM**self.ACTION_DIM

    def get_action(self, s_t):
        actions = []
        for picker_i in range(len(self.env.pickers_speed)):
            action = self._get_action_for_picker_i(picker_i, s_t)
            actions.append(action)

        actions = np.asarray(actions, dtype=np.int32)

        return actions

    def _get_action_for_picker_i(self, picker_i, s_t):
        fruit_map = s_t[3]
        picker_x_pixel = self.env.platform_x_pixel + picker_i * self.env.PICKER_DISTANCE + self.env.PICKING_WINDOWS_SIZE

        pic_win_x_low = picker_x_pixel - self.env.PICKING_WINDOWS_SIZE
        pic_win_x_high = picker_x_pixel +  self.env.PICKING_WINDOWS_SIZE

        best_picker_y_pixel = None
        best_fruit_in_pick_area = 0
        for y in range(self.env.MAP_HEIGHT - 2*self.env.PICKING_WINDOWS_SIZE):
            picker_y_pixel = y + self.env.PICKING_WINDOWS_SIZE
            pic_win_y_low = picker_y_pixel - self.env.PICKING_WINDOWS_SIZE
            pic_win_y_high = picker_y_pixel + self.env.PICKING_WINDOWS_SIZE

            pick_area = fruit_map[pic_win_y_low:pic_win_y_high+1, pic_win_x_low:pic_win_x_high+1]
            fruit_in_pick_area = np.sum(pick_area)
            if fruit_in_pick_area >= best_fruit_in_pick_area:
                best_picker_y_pixel = picker_y_pixel
                best_fruit_in_pick_area = fruit_in_pick_area
            # print(picker_y_pixel, ":", pic_win_y_low, pic_win_y_high, fruit_in_pick_area)

        current_picker_y_pixel = self.env.pickers_y_pixel[picker_i]

        if best_picker_y_pixel > current_picker_y_pixel:
            action = 2 # move down
        elif best_picker_y_pixel < current_picker_y_pixel:

            action = 1 # move up
        else:
            action = 0 # keep still


        # print("best_picker_y_pixel: ", best_picker_y_pixel)
        # print("current_picker_y_pixel: ", current_picker_y_pixel)
        # print("best_fruit_in_pick_area: ", best_fruit_in_pick_area)
        # print("action: ", action)

        return action