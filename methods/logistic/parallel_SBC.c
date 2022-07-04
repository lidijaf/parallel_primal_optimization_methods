/*
 * QuadraticMain.c
 *
 *  Created on: Oct 10, 2016
 *      Author: lidija
 */
#include "util.utility.h"
#include "io/read_write.h"
#include "neighbourhood/neighbourhood_check.h"
#include "compute_updates/comp_upd.h"

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


void DQNParallel_SBC(int N, int Dim, int size_row, int size_col, char* type, char* path){
	int my_rank;
	int procs;
	int N_w=size_col;
	int N_f, rem=0;
	int Nx=size_col+1;
	double *Adata;
	double *Bdata;

	double *WMatrix=calloc(N*N, sizeof(double));
	double *myWMatrix=calloc(N, sizeof(double));

	double stepSize=0.001;
	int iter=250000;

	int i, j, k, l, g, row, wrow, col, h;

	double lambda_penal=0.03;
	double muConst=lambda_penal;
	double epsilon=0.01;

	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	MPI_Barrier(MPI_COMM_WORLD);
	double start=MPI_Wtime();

	if (my_rank==0){ 
		find_optimal_distribution(&N_f, &rem, size_row, N);
	} 
	MPI_Bcast(&N_f, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rem, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(my_rank==0){
		Adata=calloc((N_f+rem)*N_w,sizeof(double));
		Bdata=calloc(N_f+rem, sizeof(double));
	}
	else{
		Adata=calloc(N_f*N_w,sizeof(double));
		Bdata=calloc(N_f, sizeof(double));
	}

	get_my_data(Adata, Bdata, WMatrix, myWMatrix, &stepSize, my_rank, N, Dim, N_f, N_w, Nx, lambda_penal, &rem, type, path);

	int *my_neighbours=calloc(N, sizeof(int));
	int my_neighbours_count=0;
	for(i=0;i<N;++i){
		if(myWMatrix[i]!=0.0 && my_rank!=i){
			my_neighbours[my_neighbours_count]=i;
			my_neighbours_count++;
		}
	}

	MPI_Group worldGroup, myGroup;
	MPI_Comm *allComms=calloc(N, sizeof(MPI_Comm));
	MPI_Comm myComm, tmpComm;

	for(i=0;i<N;i++){
		int numOfNeighbours;
		if(my_rank==i){
			numOfNeighbours=my_neighbours_count;
		}
		MPI_Bcast(&numOfNeighbours, 1, MPI_INT, i, MPI_COMM_WORLD);
		int *arrayOfNeighbours=calloc(numOfNeighbours+1, sizeof(int));
		if(my_rank==i){
			arrayOfNeighbours[0]=my_rank;
			for(int k=0;k<my_neighbours_count;k++)
				arrayOfNeighbours[k+1]=my_neighbours[k];
		}
		MPI_Bcast(arrayOfNeighbours, numOfNeighbours+1, MPI_INT, i, MPI_COMM_WORLD);
		MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
		MPI_Group_incl(worldGroup, numOfNeighbours+1, arrayOfNeighbours, &myGroup);
		MPI_Comm_create(MPI_COMM_WORLD, myGroup, &tmpComm);

		allComms[i]=tmpComm;
		free(arrayOfNeighbours);

		if(my_rank==i)
			myComm=tmpComm;
	}

	double *X=calloc(Dim, sizeof(double));
	double *Wii=calloc(Dim*Dim, sizeof(double));
	for(i=Dim-1;i>=0;--i)
		Wii[i*Dim+i]=myWMatrix[my_rank];

	double *Gradient=calloc(Dim, sizeof(double));
	double *Xremote=calloc(Dim*(my_neighbours_count+1), sizeof(double));

	double *AWeightInv=calloc(Dim*Dim, sizeof(double));
	double *GMatrix=calloc(Dim*Dim, sizeof(double));
	double *ww = calloc(Dim - 1, sizeof(double));
	double *subMatr=calloc(Nx, sizeof(double));
	double vv, dot, coeff, coeff2;
	double *Xdiff=calloc(Dim, sizeof(double));

	double *eye = calloc(Dim * Dim, sizeof(double));
	for (h = Dim-1; h >=0; h--)
		eye[h * Dim + h] = 1.0;

	double *GradOld=calloc(Dim, sizeof(double));

	int stop=-1;
	int stopGlobal=-1;

	for(int k=iter;k>0;--k){
		
		MPI_Allreduce(&stop, &stopGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);	
		if(stopGlobal>=0)
			MPI_Bcast(&stopGlobal, 1, MPI_INT, stopGlobal, MPI_COMM_WORLD);	
		
		if(stopGlobal>=0){
		  break;
		}

		double *NablaPsi=calloc(Dim, sizeof(double));
		double *AWeight=calloc(Dim*Dim, sizeof(double));
		double *sDirection=calloc(Dim, sizeof(double));
		int *pivotArray=calloc(Dim, sizeof(int));

		compute_Gradient(N_w, N_x, N_f, rem, Dim, X, Gradient, GradOld, my_rank, Adata, Bdata, lambda_penal);

		compute_Hessian(Dim, N_f, N_w, N_x, my_rank, rem, Adata, ww, vv, lambda_penal, GMatrix, eye);

			for(int c=0;c<N;c++){
				if(my_rank==c || is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count)){
					MPI_Gather(X, Dim, MPI_DOUBLE, Xremote, Dim, MPI_DOUBLE, 0, allComms[c]);
				}
			}

			for(i=0;i<my_neighbours_count;++i){
		    		double *zero=calloc(Dim, sizeof(double));
		    		int my_neighbours_rank=my_neighbours[i];
	   	    	LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, Xdiff, Dim);
	   	    	cblas_daxpy(Dim, -1, Xremote+(i+1)*Dim, 1, Xdiff, 1);

		   	    cblas_daxpy(Dim, myWMatrix[my_neighbours_rank], Xdiff, 1, zero, 1);
		   	    cblas_daxpy(Dim, 1, zero, 1, NablaPsi, 1);
	
			    	free(zero);
	   	}
	    	
	   	double *curRes=calloc(Dim, sizeof(double));
	   	double *GradientGlob=calloc(Dim*N, sizeof(double));
	 
	   	for(i=0;i<Dim;i++){
	   		curRes[i]=(1-myWMatrix[my_rank])*X[i];
	   		for(j=1;j<=my_neighbours_count;j++){
	   			int my_neighbours_rank=my_neighbours[j-1];
	   			curRes[i]+=-myWMatrix[my_neighbours_rank]*Xremote[j*Dim+i];
	   		}
	   		curRes[i]+=stepSize*GradOld[i];
	   	}

	   	MPI_Gather(curRes, Dim, MPI_DOUBLE, GradientGlob, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

		  double euclidean_norm;
		  if(my_rank==0){
			  euclidean_norm=cblas_dnrm2 (Dim*N, GradientGlob, 1);
		  }
	  	MPI_Bcast(&euclidean_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		  if(k<iter && euclidean_norm<epsilon && euclidean_norm>0.0){
			  stop=my_rank;
			  continue;
		}

	  cblas_daxpy(Dim, stepSize, Gradient, 1, NablaPsi, 1);

	  for(i=Dim-1;i>=0;--i)
	    AWeight[i*Dim+i]=1.0;

	  cblas_daxpy(Dim*Dim, stepSize, GMatrix, 1, AWeight, 1);
	  cblas_daxpy(Dim*Dim, -1, Wii, 1, AWeight, 1);

	  LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, AWeight, Dim, AWeightInv, Dim);
	  LAPACKE_dgetrf(LAPACK_ROW_MAJOR, Dim, Dim, AWeightInv, Dim, pivotArray);
	  LAPACKE_dgetri( LAPACK_ROW_MAJOR, Dim, AWeightInv, Dim, pivotArray);

	  cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1.0, AWeightInv, Dim, NablaPsi, 1, 1.0, sDirection, 1);
	  cblas_daxpy(Dim, -1, sDirection, 1, X, 1);
		
		free(NablaPsi);
		free(AWeight);
		free(sDirection);
		free(pivotArray);
	}
	
	free(Gradient);
	free(Xremote);
	free(AWeightInv);
	free(GMatrix);
	free(ww);
	free(subMatr);
	free(eye);
	free(Xdiff);

	free(Adata);
	free(Bdata);
	free(WMatrix);
	free(myWMatrix);
	free(Wii);
	free(my_neighbours);
	free(allComms);

	double end=MPI_Wtime();
	double elapsed=end-start;
	double max_time;
	MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	double *Xall=calloc(Dim*N, sizeof(double));
	MPI_Gather(X, Dim, MPI_DOUBLE, Xall, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(my_rank==0){
		FILE *file;
		char fileName[50];
		strcpy(fileName, path);
		strcat(fileName, "/");
		strcat(fileName, type);
		strcat(fileName,"log_SBC.txt");
		printf("%s\n", fileName);
		file=fopen(fileName,"w");
		fprintf(file, "N=%d, Dim=%d, rows=%d, cols=%d\n", N, Dim, size_row, size_col);
		printf("\nTotal time: %f\n", max_time);

	}
	free(X);
	MPI_Finalize();
}
