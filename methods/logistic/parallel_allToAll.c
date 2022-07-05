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
#include <math.h>

double max(double* a, int Dim);
double min(double* a, int Dim);
void printVector(double* vector, int n, char* name, int id);
void DQNquadraticParallel(int N, int Dim, int size_row, int size_col);


int main(int argc ,char* argv[]) {

	if(argc<5){
		printf("Input parameters N, Dim, size_row and size_col not specified.");
		return -1;
	}
	int N=atoi(argv[1]);
	int Dim=atoi(argv[2]);
	int size_row=atoi(argv[3]);
	int size_col=atoi(argv[4]);
	DQNquadraticParallel(N, Dim, size_row, size_col);
	return 0;
}

void get_my_data(double* A, double *B, double *WMatrix, double *stepSize, int my_rank, int N, int Dim, int N_f, int N_w, int Nx, double lambda_penal,
					double *LConst, double *wMin, double *wMax, int* rem){

	
	double *Adata=calloc(N_f*N_w*N+*rem*N_w, sizeof(double));
	double *Bdata=calloc(N_f*N+*rem, sizeof(double));
	int *Adj=calloc(N*N, sizeof(double));
	int *degreeSensor=calloc(N, sizeof(double));

	if(my_rank==0){
		int fd;
		char *infile = "Adata.bin";
		int bytes_expected=(N_f*N_w*N+*rem*N_w)*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Adata, bytes_expected);

		infile = "Bdata.bin";
		bytes_expected=(N_f*N+*rem)*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Bdata, bytes_expected);

		infile="Adj.bin";
		bytes_expected = N * N * sizeof(int);
		fd=open(infile, O_RDONLY);
		read(fd, Adj, bytes_expected);

		infile="degSens.bin";
		bytes_expected = N * sizeof(int);
		fd=open(infile, O_RDONLY);
		read(fd, degreeSensor, bytes_expected);
	}
	

	if(*rem!=0){

		int *sendcounts=calloc(N, sizeof(int));
		int *displs=calloc(N, sizeof(int));
		sendcounts[0]=N_w*(N_f+*rem);
		displs[0]=0;
		for(int p=1;p<N;p++){
			sendcounts[p]=N_w*N_f;
			displs[p]=p*N_w*N_f+N_w**rem;		
		}


		MPI_Scatterv(Adata, sendcounts, displs, MPI_DOUBLE, A, sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	
		sendcounts[0]=N_f+*rem;
		displs[0]=0;
		for(int p=1;p<N;p++){
			sendcounts[p]=N_f;
			displs[p]=N_f*p+*rem;
		}
		

		MPI_Scatterv(Bdata, sendcounts, displs, MPI_DOUBLE, B, sendcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	}
	else{
		MPI_Scatter(Adata, N_f*N_w, MPI_DOUBLE, A, N_f*N_w, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatter(Bdata, N_f, MPI_DOUBLE, B, N_f, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}

	free(Adata);
	free(Bdata);


	double LTempLocal = 0.0;

	double *temp_matrix=calloc(Nx*Nx, sizeof(double));
	for(int l=0;l<N_f+(my_rank==0)**rem;l++){
		double *subMatr = calloc(Nx, sizeof(double));
		double *tmp=calloc(Nx*Nx, sizeof(double));
		for (int h = 0; h < Nx-1; h++)
			subMatr[h] = A[l * N_w+h] * B[l];
		subMatr[Nx-1] = B[l];
		cblas_dger(CblasRowMajor, Nx, Nx, 1.0, subMatr, 1, subMatr, 1, tmp, Nx);
		cblas_daxpy(Nx*Nx, 1, tmp, 1, temp_matrix, 1);
		free(tmp);
		free(subMatr);

	double *TempMatrix=calloc(Nx*Nx*N, sizeof(double));
	MPI_Gather(temp_matrix, Nx*Nx, MPI_DOUBLE, TempMatrix, Nx*Nx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if(my_rank==0){
		for(int u=1;u<N;u++){
			cblas_daxpy(Nx*Nx, 1, TempMatrix+Nx*Nx*u, 1, temp_matrix, 1);
		}

		double wr[Nx], wi[Nx], vl[Nx * Nx], vr[Nx * Nx];
		LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'N', 'N', Nx, temp_matrix, Nx, wr, wi, vl, Nx, vr, Nx);
		double norm=LTempLocal = max(wr, Nx);
		printf("norm=%.3f", norm);

		*LConst=1.0/(4.0*N) * norm + lambda_penal;
		*stepSize=1.0/(*LConst)/100.0;
		printf("stepSize here = %.15f", *stepSize);
	}
	MPI_Bcast(stepSize, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(temp_matrix);
	
	free(TempMatrix);

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

		double *WMdiag=calloc(N, sizeof(double));
		for(int j=0;j<N;j++)
			WMdiag[j]=WMatrix[j*N+j];
		*wMin=min(WMdiag, N);
		*wMax=max(WMdiag, N);

		free(eye);
		free(eyeScaled);
		free(diagMatr);
		free(v);
		free(WMdiag);
	}
	MPI_Bcast(WMatrix, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(wMin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(wMax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	
	free(Adj);
	free(degreeSensor);

}

void find_optimal_distribution(int *N_f, int *rem, int size_row, int N){
	if(size_row % N ==0){
		*N_f=size_row%N;
		*rem=0;
	}
	else{
		*N_f=size_row / N;
		*rem=size_row % N;
	}
}

void DQNquadraticParallel(int N, int Dim, int size_row, int size_col){
	int my_rank;
	int procs;
	int N_w=size_col;
	int N_f, rem=0;
	int Nx=size_col+1;
	double *Adata;
	double *Bdata;

	double *WMatrix=calloc(N*N, sizeof(double));
	double stepSize=0.0;
	int i, j, k, l, g, row, wrow, col, h;

	double lambda_penal=0.03;
	double wMin, wMax;

	double muConst=lambda_penal;
	double LConst;

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

	get_my_data(Adata, Bdata, WMatrix, &stepSize, my_rank, N, Dim, N_f, N_w, Nx, lambda_penal, &LConst, &wMin, &wMax, &rem);


	int *my_neighbours=calloc(N, sizeof(int));
	int my_neighbours_count=0;
	for(i=N-1;i>=0;--i){
		if(WMatrix[my_rank*N+i]!=0.0 && my_rank!=i){
			my_neighbours[my_neighbours_count]=i;
			my_neighbours_count++;
		}
	}

	double *X=calloc(Dim, sizeof(double));

	double *Wii=calloc(Dim*Dim, sizeof(double));
	for(i=Dim-1;i>=0;--i)
		Wii[i*Dim+i]=WMatrix[my_rank*N+my_rank];

	double *Gradijent=calloc(Dim, sizeof(double));
	double *Xremote=calloc(Dim*N, sizeof(double));
	double *AWeightInv=calloc(Dim*Dim, sizeof(double));
	double *GMatrix=calloc(Dim*Dim, sizeof(double));
	double *ww = calloc(Dim - 1, sizeof(double));
	double *subMatr=calloc(Nx, sizeof(double));
	double vv, dot, coeff, coeff2;
	double *Xdiff=calloc(Dim, sizeof(double));
	double *AWeightInvWh=calloc(N*Dim*N*Dim, sizeof(double));
	double *NablaPsiWh=calloc(Dim*N, sizeof(double));
	double *NablaDvaPsi=calloc(N*Dim*Dim, sizeof(double));
	double *VMatrix=calloc(Dim*Dim*N, sizeof(double));
	double *uVektorExt=calloc(N*Dim, sizeof(double));
  	double *WuWh=calloc(N*Dim*N*Dim, sizeof(double));

	double *eye = calloc(Dim * Dim, sizeof(double));
	for (h = Dim-1; h >=0; h--)
		eye[h * Dim + h] = 1.0;

	double *WKronecker=calloc(Dim*Dim*N*N, sizeof(double));
	int rowKron=0, rowEye=0;
	for(row=0;row<Dim*N;row++){
	 	for(wrow=0;wrow<N;wrow++){
	    		for(col=0;col<Dim;col++){
	    			WKronecker[row*N*Dim+wrow*Dim+col]=WMatrix[rowKron*N+wrow]*eye[rowEye*Dim+col];
	    		}
	    	}
		if((row+1)%Dim==0)
			rowKron+=1;
		rowEye+=1;
		if(rowEye==Dim)
			rowEye=0;
	 }

	 double *WDiag=calloc(Dim*Dim*N*N, sizeof(double));
	 for(l=0;l<Dim*N;l++){
	 	WDiag[l*N*Dim+l]=WKronecker[l*N*Dim+l];
	 }

	 double *Wu=calloc(Dim*Dim*N*N, sizeof(double));
	 LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', Dim*N, Dim*N, WKronecker, Dim*N, Wu, Dim*N);
	 cblas_daxpy(Dim*Dim*N*N, -1, WDiag, 1, Wu, 1);
	
	for(int k=2000;k>0;--k){
		
		double *NablaPsi=calloc(Dim, sizeof(double));
		double *AWeight=calloc(Dim*Dim, sizeof(double));
		double *sDirection=calloc(Dim, sizeof(double));
		int *pivotArray=calloc(Dim, sizeof(int));


		double *Sum=calloc(N_w+1, sizeof(double));
		
		vv = X[Dim - 1];
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim-1, X, Dim-1, ww, Dim-1);

		//for(l=0;l<N_f+(my_rank==0)*rem;l++){
		for(l=N_f+(my_rank==0)*rem-1; l >= 0; l--){
			dot=cblas_ddot(Dim-1, Adata+(l * (N_w)), 1, ww, 1);
			coeff =(dot + vv)* (-Bdata[l]);
			coeff2 = exp(coeff) / (1 + exp(coeff));
			
			for (h = Nx-2; h >= 0; h--)
				subMatr[h] = Adata[l*N_w+h] * (-Bdata[l]);
			subMatr[Nx-1] = -Bdata[l];
			cblas_daxpy(Nx, coeff2, subMatr, 1, Sum, 1);
		}
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, Sum, Dim, Gradijent, Dim);
		cblas_daxpy(Dim, lambda_penal, X, 1, Gradijent, 1);
		free(Sum);


		double *SumaMatrix = calloc(Dim*Dim, sizeof(double));
		//for (l = 0; l < N_f+(my_rank==0)*rem; l++) {
		for (l = N_f+(my_rank==0)*rem-1; l >= 0; l--) {
			dot=cblas_ddot(Dim-1, Adata+(l * N_w), 1, ww, 1);
			coeff = (dot + vv) * (-Bdata[l]);
			coeff2 = exp(coeff) / (1 + exp(coeff))/(1+exp(coeff));
			double *tmp=calloc(Nx*Nx, sizeof(double));
			for (h = Nx-2; h >= 0; h--)
				subMatr[h] = Adata[l * N_w+h] * (-Bdata[l]);
			subMatr[Nx-1] = (-Bdata[l]);
			cblas_dger(CblasRowMajor, Nx, Nx, 1.0, subMatr, 1, subMatr, 1, tmp, Nx);
			cblas_daxpy(Nx*Nx, coeff2, tmp, 1, SumaMatrix, 1);
			free(tmp);
		}
		
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, SumaMatrix, Dim, GMatrix, Dim);
		free(SumaMatrix);
		cblas_daxpy(Dim*Dim, lambda_penal, eye, 1, GMatrix, 1);
		

	    	MPI_Allgather(X, Dim, MPI_DOUBLE, Xremote, Dim, MPI_DOUBLE, MPI_COMM_WORLD);

	    	for(i=my_neighbours_count-1;i>=0;--i){
		    double *zero=calloc(Dim, sizeof(double));

	   	    j=my_neighbours[i];
	   	    LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, Xdiff, Dim);
	   	    cblas_daxpy(Dim, -1, Xremote+j*Dim, 1, Xdiff, 1);
	   	    cblas_daxpy(Dim, WMatrix[my_rank*N+j], Xdiff, 1, zero, 1);
	   	    cblas_daxpy(Dim, 1, zero, 1, NablaPsi, 1);

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

	    double *AWeightInvTmp=calloc(N*Dim*Dim, sizeof(double));

	    MPI_Allgather(AWeightInv, Dim*Dim, MPI_DOUBLE, AWeightInvTmp, Dim*Dim, MPI_DOUBLE, MPI_COMM_WORLD);

    	    int offset=0;
	    int j=0;
	    for(i=0;i<N*Dim;i++){
			LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim, AWeightInvTmp+j*Dim, Dim,  AWeightInvWh+i*Dim*N+offset, Dim);
			++j;
			if(i>0 && (i+1)%Dim==0)
				offset+=Dim;
	    }

	    double *uVektorPart=calloc(Dim*Dim*N, sizeof(double));
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dim, Dim*N, Dim*N, 1.0, Wu+Dim*N*my_rank*Dim, Dim*N, AWeightInvWh, Dim*N, 1.0, uVektorPart, Dim*N);
 	   
	    double *uVektor=calloc(Dim, sizeof(double));
	    MPI_Allgather(NablaPsi, Dim, MPI_DOUBLE, NablaPsiWh, Dim, MPI_DOUBLE, MPI_COMM_WORLD);
	    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim*N, 1.0, uVektorPart, Dim*N, NablaPsiWh, 1, 1.0, uVektor, 1);

	    double *NablaDvaF=calloc(Dim*Dim*N, sizeof(double));
	    for(i=0;i<Dim;i++)
	    	LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim, GMatrix+i*Dim, Dim, NablaDvaF+i*Dim*N+my_rank*Dim, Dim);
	    double * eyeNabla=calloc(N*Dim*Dim, sizeof(double));
	    for(i=0;i<Dim;i++)
	    	eyeNabla[my_rank*Dim+i*Dim*N+i]=1.0;

	    LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', Dim, Dim*N, eyeNabla, Dim*N, NablaDvaPsi, Dim*N);
	    cblas_daxpy(Dim*Dim*N, stepSize, NablaDvaF, 1, NablaDvaPsi, 1);
	    cblas_daxpy(Dim*Dim*N, -1.0, WKronecker+my_rank*Dim*Dim*N, 1, NablaDvaPsi, 1);

	    double *BMatrix=calloc(Dim*Dim*N, sizeof(double));
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dim, Dim*N, Dim*N, 1.0, NablaDvaPsi, Dim*N, AWeightInvWh, Dim*N, 1.0, BMatrix, Dim*N);

	    LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', Dim, Dim*N, WKronecker+my_rank*Dim*Dim*N, Dim*N, VMatrix, Dim*N);
	    cblas_daxpy(Dim*Dim*N, -stepSize, NablaDvaF, 1, VMatrix, 1);

	    MPI_Allgather(uVektor, Dim, MPI_DOUBLE, uVektorExt, Dim, MPI_DOUBLE, MPI_COMM_WORLD);

	    double *Temp1=calloc(Dim, sizeof(double));
	    cblas_daxpy(Dim*Dim*N, 1.0, eyeNabla, 1, VMatrix, 1);
	    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim*N, -1.0, VMatrix, Dim*N, uVektorExt, 1, 1.0, Temp1, 1 );

	    double *LambdaMatrix=calloc(Dim*Dim*N, sizeof(double));
	    double *LambdaVector=calloc(Dim, sizeof(double));

	    if(k==2000){
	    	for(i=0;i<Dim;i++)
	    		LambdaVector[i]=Temp1[i]/uVektor[i];
	    	for(i=0;i<Dim;i++)
	    		LambdaMatrix[my_rank*Dim+i*Dim*N+i]=LambdaVector[i];
	    }

	    double  LambdaBound = 1/((stepSize*LConst)+1-wMin)*((stepSize*muConst)+1-wMax) / (1-wMin);

	    for(i=0;i<Dim;i++){
	    	if(LambdaMatrix[my_rank*Dim+i*Dim*N+i]< -LambdaBound)
	    		LambdaMatrix[my_rank*Dim+i*Dim*N+i]= -LambdaBound;
	    	else if(LambdaMatrix[my_rank*Dim+i*Dim*N+i]+LambdaBound)
	    		LambdaMatrix[my_rank*Dim+i*Dim*N+i]= -LambdaBound;
	    }

	    double *Prod=calloc(N*Dim*Dim, sizeof(double));
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dim, Dim*N, Dim*N, 1.0, LambdaMatrix, Dim*N, Wu, Dim*N, 1.0, Prod, Dim*N);
	    cblas_daxpy(Dim*Dim*N, -1.0, Prod, 1, eyeNabla, 1 );

	    double *Prod1=calloc(Dim*Dim*N, sizeof(double));
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dim, Dim*N, Dim*N, -1.0, eyeNabla, Dim*N, AWeightInvWh, Dim*N, 1.0, Prod1, Dim*N);

	    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim*N, 1.0, Prod1, Dim*N, NablaPsiWh, 1, 1.0, sDirection, 1);

	    cblas_daxpy(Dim, 1, sDirection, 1, X, 1);


		free(NablaPsi);
		free(AWeight);
		free(sDirection);
		free(pivotArray);
		
		free(AWeightInvTmp);
		free(uVektorPart);
		free(uVektor);
		free(NablaDvaF);
		free(eyeNabla);
		free(BMatrix);

		free(Prod);
		free(Prod1);
		free(LambdaMatrix);
		free(LambdaVector);
		free(Temp1);

	}
	
	free(Gradijent);
	free(Xremote);
	free(WKronecker);	
	free(AWeightInv);
	free(GMatrix);
	free(ww);
	free(subMatr);
	free(eye);
	free(Xdiff);
	free(WDiag);
	free(Wu);
	free(AWeightInvWh);
	free(NablaPsiWh);
	free(NablaDvaPsi);
	free(VMatrix);
	free(uVektorExt);
	free(WuWh);

	free(Adata);
	free(Bdata);
	free(WMatrix);
	free(Wii);

	free(my_neighbours);

	double end=MPI_Wtime();
	double elapsed=end-start;
	double max_time;
	MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(my_rank==0){
		printf("\nTotal time: %f\n", max_time);
		printVector(X, Dim, "\n The result is", my_rank);
	}


	free(X);
	MPI_Finalize();

}


void printVector(double* vector, int n, char* name, int id){
	printf("\n%d Printing %s\n", id, name);
	int i;
	for(i=0;i<n;++i){
		printf("  %8.2f  ", vector[i]);
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
