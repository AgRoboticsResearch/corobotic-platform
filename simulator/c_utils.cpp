#include <stdlib.h> 
#include <algorithm>
#include <time.h> 
#include <iostream>

extern "C"
{

bool c_check_empty(int* matrix, int matrix_w, int matrix_h){

   bool empty = true;

   for(int i = 0; i < matrix_w*matrix_h; i++ )
   {

        if(matrix[i] != 0)
        {
            empty = false;
            break;
        }
   }

   return empty;
}

bool c_check_empty_large(int* fruit_map, int fruit_map_x, int fruit_map_y, int pic_win_y_low, int pic_win_y_high, int pic_win_x_low, int pic_win_x_high){

   bool empty = true;

   for(int x = pic_win_x_low; x < pic_win_x_high; x++ )
   {

       for(int y = pic_win_y_low; y < pic_win_y_high; y++ )
       {
            if(fruit_map[x * fruit_map_y + y] != 0)
            {
                empty = false;
                break;
            }
        }
   }

   return empty;
}


bool c_pick_random_n_opt(int* fruit_map, int fruit_map_x, int fruit_map_y, int n, int* n_picked, int pic_win_y_low, int pic_win_y_high, int pic_win_x_low, int pic_win_x_high, int seed){

    srand(seed);

    bool empty = false;

    int rand_x, rand_y, i;

    int area_x = pic_win_x_high - pic_win_x_low;
    int area_y = pic_win_y_high - pic_win_y_low;

    i = 0;

    while(true){
        if(i >= n)
        {
            break;
        }

        if(c_check_empty_large(fruit_map, fruit_map_x, fruit_map_y, pic_win_y_low, pic_win_y_high, pic_win_x_low, pic_win_x_high))
        {
            empty = true;
            break;
        }

        rand_x = rand() % area_x + pic_win_x_low;        
        rand_y = rand() % area_y + pic_win_y_low;        
        
        if(fruit_map[rand_x * fruit_map_y + rand_y] > 0)
        {
            fruit_map[rand_x * fruit_map_y + rand_y] -= 1;
            i += 1;
        }

    }

    *n_picked = i;
        

    return empty;
}


bool c_pick_random_n(int* area, int area_w, int area_h, int n, int* n_picked){


    bool empty = false;
    int area_size;

    area_size = area_w * area_h;

    int idx, i;
    i = 0;

    while(true){
        if(i >= n)
        {
            break;
        }

        if(c_check_empty(area, area_w, area_h))
        {
            empty = true;
            break;
        }

        idx = rand() % area_size;        
        
        if(area[idx] > 0)
        {
            area[idx] -= 1;
            i += 1;
        }

    }
    *n_picked = i;
        

    return empty;
}

void c_move_picker(const int* action_t, int* pickers_y, const int picker_num, const int* x_limit){
    int action_i;
    for(int i = 0; i < picker_num; i++)
    {
        action_i = action_t[i];
        if(action_i == 0)
        {
            pickers_y[i] = std::max(x_limit[0], pickers_y[i] - 1 );
        }
        if(action_i == 2)
        {
            pickers_y[i] = std::min(x_limit[1], pickers_y[i] + 1 );
        }
        
    }

}

}
























