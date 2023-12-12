cdef extern from "c_utils.cpp":
    bint c_pick_random_n(int* area, int area_w, int area_h, int n, int* n_picked)
    bint c_pick_random_n_opt(int* fruit_map, int fruit_map_x, int fruit_map_y, int n, int* n_picked, int pic_win_y_low, int pic_win_y_high, int pic_win_x_low, int pic_win_x_high, int seed)
    bint c_check_empty(int* matrix, int matrix_w, int matrix_h)
    void c_move_picker(const int* action_t, int* pickers_y, const int picker_num, const int* x_limit)
