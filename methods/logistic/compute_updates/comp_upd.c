#include "comp_upd.h"	  

void compute_Gradient(int N_w, int Dim, double* X, double* Gradient, double *GradOld, int my_rank, int N_f, int rem, double *Adata, double* Bdata, int N_x, double lambda_penal){
    double *Sum=calloc(N_w+1, sizeof(double));
		double vv = X[Dim - 1];
    double *ww = calloc(Dim - 1, sizeof(double));
    double dot, coeff, coeff2;
    double *subMatr=calloc(Nx, sizeof(double));
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim-1, X, Dim-1, ww, Dim-1);
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim, Gradient, Dim, GradOld, Dim);
    
		for(l=N_f+(my_rank==0)*rem-1; l >= 0; l--){
			dot=cblas_ddot(Dim-1, Adata+(l * (N_w)), 1, ww, 1);
			coeff =(dot + vv)* (-Bdata[l]);
			coeff2 = exp(coeff) / (1 + exp(coeff));
			
			for (h = Nx-2; h >= 0; h--)
				subMatr[h] = Adata[l*N_w+h] * (-Bdata[l]);
			subMatr[Nx-1] = -Bdata[l];
			cblas_daxpy(Nx, coeff2, subMatr, 1, Sum, 1);
		}
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, Sum, Dim, Gradient, Dim);
		cblas_daxpy(Dim, lambda_penal, X, 1, Gradient, 1);
		free(Sum);
}
