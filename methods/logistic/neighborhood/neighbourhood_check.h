#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include <lapacke.h>
#include <stdio.h>
#include <lapacke_utils.h>
#include <cblas.h>
#include <mpi.h>
#include <sysexits.h>
#include <err.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include<time.h>

int is_my_neighbour(int my_rank, int i, int *my_neighbours, int my_neighbours_count);

int get_my_active_neighbour(int k, int my_rank, int *my_neighbours, int my_neighbours_count, int *active);

#endif
