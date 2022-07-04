#include "neighbourhood_check.h"

int is_my_neighbour(int my_rank, int i, int *my_neighbours, int my_neighbours_count){
	for(int j=my_neighbours_count-1;j>=0;j--)
		if(my_neighbours[j]==i)
			return 1;
	return 0;
}

int get_my_active_neighbour(int k, int my_rank, int *my_neighbours, int my_neighbours_count, int *active){
	int cnt_active=-1;
	int neighbour=-1;
	for(int i=0;i<my_neighbours_count;i++){
		if(active[my_neighbours[i]]){
			++cnt_active;
			neighbour=my_neighbours[i];
		}
		if(cnt_active==k)
			return neighbour;
	}
}
