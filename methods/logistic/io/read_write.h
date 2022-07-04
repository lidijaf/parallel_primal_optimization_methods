#ifndef UTILITY_H
#define UTILITY_H

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

void printVector(double* vector, int n, char* name, int id);
void get_my_data(double* A, double *B, double *WMatrix, double *myWMatrix, double *stepSize, int my_rank, int N, int Dim, int N_f, int N_w, int Nx, double lambda_penal,
					 int* rem, char* type, char* path);
					 
					 
#endif
