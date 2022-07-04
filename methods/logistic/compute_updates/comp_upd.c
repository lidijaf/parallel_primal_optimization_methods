#include "comp_upd.h"	  

void compute_Gradient(int N_w, int N_x, int N_f, int rem, int Dim, double* X, double* Gradient, double *GradOld, int my_rank, double *Adata, double* Bdata, double lambda_penal){
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


void compute_Hessian(int Dim, int N_f, int N_w, int N_x, int my_rank, int rem, double *Adata, double* ww, double vv, double lambda_penal, double *GMatrix, double* eye){
	double *SumMatrix = calloc(Dim*Dim, sizeof(double));
	double dot, coeff, coeff2;
	double *subMatr=calloc(Nx, sizeof(double));
	for (l = N_f+(my_rank==0)*rem-1; l >= 0; l--) {
		dot=cblas_ddot(Dim-1, Adata+(l * N_w), 1, ww, 1);
		coeff = (dot + vv) * (-Bdata[l]);
		coeff2 = exp(coeff) / (1 + exp(coeff))/(1+exp(coeff));
		double *tmp=calloc(Nx*Nx, sizeof(double));
		for (h = Nx-2; h >= 0; h--)
			subMatr[h] = Adata[l * N_w+h] * (-Bdata[l]);
		 subMatr[Nx-1] = (-Bdata[l]);
		 cblas_dger(CblasRowMajor, Nx, Nx, 1.0, subMatr, 1, subMatr, 1, tmp, Nx);
		 cblas_daxpy(Nx*Nx, coeff2, tmp, 1, SumMatrix, 1);
		 free(tmp);
	 }
	 LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, SumMatrix, Dim, GMatrix, Dim);
	 free(SumMatrix);
	 cblas_daxpy(Dim*Dim, lambda_penal, eye, 1, GMatrix, 1);	
}
