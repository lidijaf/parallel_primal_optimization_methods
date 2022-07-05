/*
 * QuadraticMain.c
 *
 *  Created on: Oct 10, 2016
 *      Author: lidija
 */
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


double max(double* a, int Dim);
double min(double* a, int Dim);
void printVector(double* vector, int n, char* name, int id);
void DQNquadraticParallel(int N, int Dim);


int main(int argc ,char* argv[]) {

	if(argc<3){
		printf("Input parameters N and Dim not specified.");
		return -1;
	}
	int N=atoi(argv[1]);
	int Dim=atoi(argv[2]);
	DQNquadraticParallel(N, Dim);
	return 0;
}

void get_my_data(double* A, double *B, double *WMatrix, double *stepSize, int my_rank, int N, int Dim){
	double *Adata=calloc(N*Dim*Dim, sizeof(double));
	double *Bdata=calloc(N*Dim, sizeof(double));
	int *Adj=calloc(N*N, sizeof(double));
	int *degreeSensor=calloc(N, sizeof(double));
	if(my_rank==0){
		int fd;
		char *infile = "Adata.bin";
		int bytes_expected=N*Dim*Dim*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Adata, bytes_expected);

		infile = "Bdata.bin";
		bytes_expected=N*Dim*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Bdata, bytes_expected);
			
		infile="Adj.bin";
		bytes_expected=N*N*sizeof(int);
		fd=open(infile, O_RDONLY);
		read(fd, Adj, bytes_expected);

		infile="degSens.bin";
		bytes_expected = N * sizeof(int);
		fd=open(infile, O_RDONLY);
		read(fd, degreeSensor, bytes_expected);
	}

	MPI_Scatter(Adata, Dim*Dim, MPI_DOUBLE, A, Dim*Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(Bdata, Dim, MPI_DOUBLE, B, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double LTempLocal = 0.0;
	double *AdataCopy=calloc(Dim*Dim, sizeof(double));
	LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, Adata, Dim, AdataCopy, Dim);
	double *wr=calloc(Dim, sizeof(double)), *wi=calloc(Dim,sizeof(double)), *vl=calloc(Dim * Dim,sizeof(double)), *vr=calloc(Dim * Dim,sizeof(double));
	LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'N', 'N', Dim, AdataCopy, Dim, wr, wi, vl, Dim, vr, Dim);
	LTempLocal = max(wr, Dim);
	double LConst=0.0;
	MPI_Allreduce(&LTempLocal, &LConst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	*stepSize = 1.0 / LConst / 200.0;
	free(wr);free(wi);free(vl);free(vr);
	int i,j;
	if(my_rank==0){
		for(i=N-2;i>=0;--i){
			for(j=N-1;j>=i+1;--j){
				if(Adj[i*N+j]==1){
					WMatrix[i*N+j] = 1.0 / (1.0 + MAX(degreeSensor[i], degreeSensor[j]));
					WMatrix[j*N+i] = WMatrix[i*N+j];
				}
			}
		}
		
		double *eye=calloc(N*N,sizeof(double));
		double ones[N];
		for(i=N-1;i>=0;--i){
			eye[i*N+i]=1.0;
			ones[i]=1.0;
		}
			
		double *v=calloc(N, sizeof(double));
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1, WMatrix, N, ones, 1, 0, v, 1);
		double *diagMatr=calloc(N*N, sizeof(double));
		for(i=N-1;i>=0;--i)
			diagMatr[i*N+i]=v[i];
		cblas_daxpy(N*N, -1,diagMatr, 1, eye, 1);
		cblas_daxpy(N*N, 1,eye, 1, WMatrix, 1);

		double *eyeScaled=calloc(N*N, sizeof(double));
		for(i=N-1;i>=0;--i){
			eyeScaled[i*N+i]=0.5;
		}
		for(i=N*N-1;i>=0;--i)
			WMatrix[i]*=0.5;

		cblas_daxpy(N*N, 1, eyeScaled, 1, WMatrix, 1);
		free(eye);
		free(eyeScaled);
		free(diagMatr);
		free(v);
	}
	MPI_Bcast(WMatrix, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	free(Adata);
	free(Bdata);
	free(Adj);
	free(degreeSensor);
	free(AdataCopy);
	
}
void DQNquadraticParallel(int N, int Dim){
	int my_rank;
	int procs;
	double *Adata=calloc(Dim*Dim,sizeof(double));
	double *Bdata=calloc(Dim, sizeof(double));
	double *WMatrix=calloc(N*N, sizeof(double));
	double stepSize=0.0;
	int i;

	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

	MPI_Barrier(MPI_COMM_WORLD);
	double start=MPI_Wtime();
	get_my_data(Adata, Bdata, WMatrix, &stepSize, my_rank, N, Dim);

	int *my_neighbours=calloc(N, sizeof(int));
	int my_neighbours_count=0;
	for(i=N-1;i>=0;--i){
		if(WMatrix[my_rank*N+i]!=0.0 && my_rank!=i){
			my_neighbours[my_neighbours_count]=i;
			my_neighbours_count++;
		}
	}

	double *X=calloc(Dim, sizeof(double));
	double *Gradijent=calloc(Dim, sizeof(double));
	double *Wii=calloc(Dim*Dim, sizeof(double));
	for(i=Dim-1;i>=0;--i)
		Wii[i*Dim+i]=WMatrix[my_rank*N+my_rank];

	for(int k=2000;k>0;--k){
		double *NablaPsi=calloc(Dim, sizeof(double));
		double *Xremote=calloc(Dim*N, sizeof(double));
		double *AWeight=calloc(Dim*Dim, sizeof(double));
		double *AWeightInv=calloc(Dim*Dim, sizeof(double));
		double *sDirection=calloc(Dim, sizeof(double));
		double *GMatrix=calloc(Dim*Dim, sizeof(double));
		double *BdataCopy=calloc(Dim, sizeof(double));
		double *copyX=calloc(Dim, sizeof(double));
		int *pivotArray=calloc(Dim, sizeof(int));
		
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, copyX, Dim);
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, Bdata, Dim, BdataCopy, Dim);
		cblas_daxpy(Dim, -1, BdataCopy, 1, copyX, 1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Adata, Dim, copyX, 1, 0, Gradijent, 1);
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, Adata, Dim, GMatrix, Dim);		

	    	MPI_Allgather(X, Dim, MPI_DOUBLE, Xremote, Dim, MPI_DOUBLE, MPI_COMM_WORLD);
		int j;
	    	for(i=my_neighbours_count-1;i>=0;--i){
		    double *Xdiff=calloc(Dim, sizeof(double));
		    double *zero=calloc(Dim, sizeof(double));

	   	    j=my_neighbours[i];
	   	    LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, Xdiff, Dim);
	   	    cblas_daxpy(Dim, -1, Xremote+j*Dim, 1, Xdiff, 1);
	   	    cblas_daxpy(Dim, WMatrix[my_rank*N+j], Xdiff, 1, zero, 1);
	   	    cblas_daxpy(Dim, 1, zero, 1, NablaPsi, 1);

		    free(Xdiff);
		    free(zero);
	   	}
	    	cblas_daxpy(Dim, stepSize, Gradijent, 1, NablaPsi, 1);

	    	for(i=Dim-1;i>=0;--i)
	    		AWeight[i*Dim+i]=1.0;	

		cblas_daxpy(Dim*Dim, stepSize, GMatrix, 1, AWeight, 1);
	   	cblas_daxpy(Dim*Dim, -1, Wii, 1, AWeight, 1);

	   	LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, AWeight, Dim, AWeightInv, Dim);
	   	LAPACKE_dgetrf(LAPACK_ROW_MAJOR, Dim, Dim, AWeightInv, Dim, pivotArray);
	    	LAPACKE_dgetri( LAPACK_ROW_MAJOR, Dim, AWeightInv, Dim, pivotArray);

	    	cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1.0, AWeightInv, Dim, NablaPsi, 1, 1.0, sDirection, 1);
	    	cblas_daxpy(Dim, -1, sDirection, 1, X, 1);

		if(k==2000 && my_rank==0)

		free(NablaPsi);
		free(Xremote);
		free(AWeight);
		free(AWeightInv);
		free(sDirection);
		free(GMatrix);
		free(BdataCopy);
		free(copyX);
		free(pivotArray);	

	}
		
	free(Adata);
	free(Bdata);
	free(WMatrix);
	free(Wii);
	free(Gradijent);
	free(my_neighbours);

	double end=MPI_Wtime();
	double elapsed=end-start;
	double max_time;
	MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank==0){
		printf("\nTotal time: %f\n", max_time);
		//printVector(X, Dim, "\n The result is", my_rank);
	}
	double* Xglobal=calloc(N*Dim,sizeof(double));
	MPI_Gather(X, Dim, MPI_DOUBLE, Xglobal, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	if(my_rank==0)	
		printVector(Xglobal, Dim*N, "\nThe global result is ", my_rank);
	free(X);
	free(Xglobal);
	MPI_Finalize();
}


void printVector(double* vector, int n, char* name, int id){
	printf("\n%d Printing %s\n", id, name);
	int i;
	for(i=0;i<n;++i){
		printf("  %20.14f  ", vector[i]);	
	}
	printf("\n");
}

double min(double* a, int Dim) {
	double minVal = a[0];
	int j;
	for (j = Dim-1; j >0; --j)
		minVal = MIN(minVal, a[j]);
	return minVal;
}

double max(double* a, int Dim) {
	double maxVal = a[0];
	int j;
	for (j = Dim-1; j >0; --j)
		maxVal = MAX(maxVal, a[j]);
	return maxVal;
}
