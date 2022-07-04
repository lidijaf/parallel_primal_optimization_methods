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
#include <stdlib.h>
#include<time.h>

double max(double* a, int Dim);
double min(double* a, int Dim);
int minInt(int* a, int Dim);
void printVector(double* vector, int n, char* name, int id);
void DQNquadraticParallel(int N, int Dim, int size_row, int size_col, char* type, char* path);
int get_my_active_neighbour(int k, int my_rank, int *my_neighbours, int my_neighbours_count, int *active);

int main(int argc ,char* argv[]) {
	if(argc<7){
		printf("Input parameters N, Dim, size_row and size_col not specified.");
		return -1;
	}
	int N=atoi(argv[1]);
	int Dim=atoi(argv[2]);
	int size_row=atoi(argv[3]);
	int size_col=atoi(argv[4]);
	char* type=argv[5];
	char* path=argv[6];
	DQNquadraticParallel(N, Dim, size_row, size_col, type, path);
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

void get_my_data(double* A, double *B, double *WMatrix, double *myWMatrix, double *stepSize, int my_rank, int N, int Dim, int N_f, int N_w, int Nx, double lambda_penal,
					 int* rem, char* type, char* path){
	double *Adata=calloc(N_f*N_w*N+*rem*N_w, sizeof(double));
	double *Bdata=calloc(N_f*N+*rem, sizeof(double));
	int *Adj=calloc(N*N, sizeof(double));
	int *degreeSensor=calloc(N, sizeof(double));

	if(my_rank==0){
		int fd;
		char infile[50];
		strcpy(infile, type);
		strcat(infile, "/Adata.bin");
		printf("%s\n", infile);
		int bytes_expected=(N_f*N_w*N+*rem*N_w)*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Adata, bytes_expected);

		strcpy(infile, type);
		strcat(infile, "/Bdata.bin");
		printf("%s\n", infile);
		bytes_expected=(N_f*N+*rem)*sizeof(double);
		fd=open(infile, O_RDONLY);
		read(fd, Bdata, bytes_expected);

		strcpy(infile, path);
		strcat(infile, "/Adj.bin");
		printf("%s\n", infile);
		bytes_expected = N * N * sizeof(int);
		fd=open(infile, O_RDONLY);
		read(fd, Adj, bytes_expected);

		strcpy(infile, path);
		strcat(infile, "/degSens.bin");
		printf("%s\n", infile);
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

		free(eye);
		free(eyeScaled);
		free(diagMatr);
		free(v);
		free(WMdiag);
	}
	MPI_Scatter(WMatrix, N, MPI_DOUBLE, myWMatrix, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(Adj);
	free(degreeSensor);
}

void find_optimal_distribution(int *N_f, int *rem, int size_row, int N){
	if(size_row % N ==0){
		*N_f=size_row/N;
		*rem=0;
	}
	else{
		*N_f=size_row / N;
		*rem=size_row % N;
	}

}

int is_my_neighbour(int my_rank, int i, int *my_neighbours, int my_neighbours_count){
	for(int j=my_neighbours_count-1;j>=0;j--)
		if(my_neighbours[j]==i)
			return 1;
	return 0;
}

void DQNquadraticParallel(int N, int Dim, int size_row, int size_col, char* type, char* path){
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
	int stopGlobal=-1;

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

	double *Gradijent=calloc(Dim, sizeof(double));

	double *AWeightInv=calloc(Dim*Dim, sizeof(double));
	double *GMatrix=calloc(Dim*Dim, sizeof(double));
	double *ww = calloc(Dim - 1, sizeof(double));
	double *subMatr=calloc(Nx, sizeof(double));
	double vv, dot, coeff, coeff2;
	double *Xdiff=calloc(Dim, sizeof(double));

	double *eye = calloc(Dim * Dim, sizeof(double));
	for (h = Dim-1; h >=0; h--)
		eye[h * Dim + h] = 1.0;

	for(int i=0;i<Dim;i++)
	   	AWeightInv[i*Dim+i] = 1.0;

	double *GradOld=calloc(Dim, sizeof(double));

	double local_communication_time_end, local_communication_time_start, local_comm_time=0.0;
	int iters=0;

	int stop=-1;
	int *active=calloc(N, sizeof(int));
	MPI_Comm *allCurrComms;

	int probab_bound=0;

	for(int k=iter;k>0;--k){

		MPI_Barrier(MPI_COMM_WORLD);

		//free the communicators from the previous iteration
		if(k<iter){
			for(int c=0;c<N;c++){
					if (my_rank==c || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[my_rank]))
						MPI_Comm_free(&allCurrComms[c]);
			}
			free(allCurrComms);
		}
		
		MPI_Allreduce(&stop, &stopGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);	
		if(stopGlobal>=0)
			MPI_Bcast(&stopGlobal, 1, MPI_INT, stopGlobal, MPI_COMM_WORLD);	
		
		if(stopGlobal>=0){
		  break;
		}
		
		probab_bound=(1.0-pow(0.5,iter-k+1))*10;
		if(probab_bound<5)
			probab_bound=5;


		active=calloc(N, sizeof(int));
		int activeLoc=0;

		srand(time(NULL)+my_rank+k);
		int comm_probab=0;
		comm_probab=rand() % 10 +1;
		int color=1;
		allCurrComms=calloc(N, sizeof(MPI_Comm));

		if(comm_probab <= probab_bound)
			activeLoc=1;

		MPI_Allgather(&activeLoc, 1, MPI_INT, active, 1, MPI_INT, MPI_COMM_WORLD);

		for(int c=0;c<N;c++){
			if(my_rank==c || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[my_rank])){
				int my_local_rank;
				MPI_Comm_rank(allComms[c], &my_local_rank);
				MPI_Comm_split(allComms[c], 1, my_local_rank, &allCurrComms[c]);
			}
			else if(is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && !active[my_rank]){
				int my_local_rank;
				MPI_Comm_rank(allComms[c], &my_local_rank);
				MPI_Comm_split(allComms[c], MPI_UNDEFINED, my_local_rank, &allCurrComms[c]);
			}
		}
		
		double *NablaPsi=calloc(Dim, sizeof(double));
		double *AWeight=calloc(Dim*Dim, sizeof(double));
		double *sDirection=calloc(Dim, sizeof(double));
		int *pivotArray=calloc(Dim, sizeof(int));

		//GRADIENT*************************************

		double *Sum=calloc(N_w+1, sizeof(double));
		vv = X[Dim - 1];
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim-1, X, Dim-1, ww, Dim-1);
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', 1, Dim, Gradijent, Dim, GradOld, Dim);

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

		//GRADIENT************************************

		//HESIAN**************************************

		double *SumaMatrix = calloc(Dim*Dim, sizeof(double));
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
		
		//HESIAN**************************************

		//EXCHANGE**************************************

		int my_communicator_size=0;
		MPI_Comm_size(allCurrComms[my_rank], &my_communicator_size);
		double *Xremote=calloc((my_communicator_size)*Dim, sizeof(double));

		for(int c=0;c<N;c++){
			if(my_rank==c || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[my_rank])){
				MPI_Gather(X, Dim, MPI_DOUBLE, Xremote, Dim, MPI_DOUBLE, 0, allCurrComms[c]);
			}
		}
	
		double active_neighbours_weight=0.0;
		for(i=0;i<my_communicator_size-1;++i){
		    	double *zero=calloc(Dim, sizeof(double));
		    	int my_neighbours_rank=get_my_active_neighbour(i, my_rank, my_neighbours, my_neighbours_count, active);
	   	    	LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, Xdiff, Dim);
	   	    	cblas_daxpy(Dim, -1, Xremote+(i+1)*Dim, 1, Xdiff, 1);

	   	    	cblas_daxpy(Dim, myWMatrix[my_neighbours_rank], Xdiff, 1, zero, 1);
			active_neighbours_weight+=myWMatrix[my_neighbours_rank];
	   	    	cblas_daxpy(Dim, 1, zero, 1, NablaPsi, 1);

		    	free(zero);
	   	}
	    	
	  //EXCHANGE************************************
	
		double *curRes=calloc(Dim, sizeof(double));
	  double *GradijentGlob=calloc(Dim*N, sizeof(double));

	   double my_self_confidence=1-active_neighbours_weight;
	 
	   	for(i=0;i<Dim;i++){
	   		curRes[i]=(1-my_self_confidence)*X[i];
	   		for(j=1;j<=my_communicator_size-1;j++){
	   			int my_neighbours_rank=get_my_active_neighbour(j-1, my_rank, my_neighbours, my_neighbours_count, active);//my_neighbours[j-1]; //ne znamo koji je ovo!!!!!!!!!!!!!!!!!!!
	   			curRes[i]+=-myWMatrix[my_neighbours_rank]*Xremote[j*Dim+i];
	   		}
	   		curRes[i]+=stepSize*GradOld[i];
	   	}

	   	MPI_Gather(curRes, Dim, MPI_DOUBLE, GradijentGlob, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);	
	
		double euclidean_norm;
		iters=0;
		if(my_rank==0){
			euclidean_norm=cblas_dnrm2 (Dim*N, GradijentGlob, 1);
			printf("%d:Euclidean norm for %d is %.5f\n", my_rank, k, euclidean_norm);
			if(k<iter && euclidean_norm<epsilon){
				iters=iter-k;
			}
		}
		MPI_Bcast(&euclidean_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(k<iter && euclidean_norm<epsilon && euclidean_norm>0.0){
			stop=my_rank;
			continue;
		}


	    	cblas_daxpy(Dim, stepSize, Gradijent, 1, NablaPsi, 1);

	     	for(i=Dim-1;i>=0;--i)
	     		AWeight[i*Dim+i]=1.0;

	     	for(i=Dim-1;i>=0;--i)
	    	 	Wii[i*Dim+i]=my_self_confidence;

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
		free(Xremote);
		free(GradijentGlob);
		free(curRes);
	
	}	

	free(Gradijent);
	free(AWeightInv);
	free(ww);
	free(subMatr);
	free(eye);
	free(Xdiff);

	free(Adata);
	free(Bdata);
	free(WMatrix);
	free(myWMatrix);
	free(my_neighbours);
	free(allComms);

	double end=MPI_Wtime();
	double elapsed=end-start;
	double max_time;
	MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	double *comm_times=calloc(N, sizeof(double));
	MPI_Gather(&local_comm_time, 1, MPI_DOUBLE, comm_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double *Xall=calloc(Dim*N, sizeof(double));
	MPI_Gather(X, Dim, MPI_DOUBLE, Xall, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(my_rank==0){
		FILE *file;
		char fileName[50];
		strcpy(fileName, path);
		strcat(fileName, "/");
		strcat(fileName, type);
		strcat(fileName,"log_.txt");
		printf("%s\n", fileName);
		file=fopen(fileName,"w");
		fprintf(file, "N=%d, Dim=%d, rows=%d, cols=%d\n", N, Dim, size_row, size_col);
		fprintf(file," Total time: %f\n", max_time);
		double avg_time=0.0;
		for(int i=0;i<N;i++){
			fprintf(file," Total communication time for process %d: %f \n", i, *(comm_times+i));
			avg_time+=*(comm_times+i);		
		}	
		avg_time=avg_time/N;
		double min_time=min(comm_times, N);
		double max_time=max(comm_times, N);
		fprintf(file,"\n Average communication time: %f \n", avg_time);
		fprintf(file," Minimal communication time: %f \n", min_time);
		fprintf(file," Maximal communication time: %f \n", max_time);
		fprintf(file," Required number of iterations: %d", iters);
	
		fclose(file);
	}
	free(X);
	MPI_Finalize();
}


void printVector(double* vector, int n, char* name, int id){
	printf("\n%d Printing %s\n", id, name);
	int i;
	for(i=0;i<n;++i){
		printf("  %8.10f  ", vector[i]);
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

int minInt(int* a, int Dim) {
	int minVal = a[0];
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
