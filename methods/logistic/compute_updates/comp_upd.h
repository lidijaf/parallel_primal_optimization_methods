#ifndef COMPUPD_H
#define COMPUPD_H

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

void compute_Gradient(int N_w, int N_x, int N_f, int rem, int Dim, double* X, double* Gradient, double *GradOld, int my_rank, double *Adata, double* Bdata, double lambda_penal);

void compute_Hessian(int Dim, int N_f, int N_w, int N_x, int my_rank, int rem, double *Adata, double* ww, double vv, double lambda_penal, double *GMatrix, double* eye);
					 
#endif
