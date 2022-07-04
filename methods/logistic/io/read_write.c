#include "read_write.h"

void printVector(double* vector, int n, char* name, int id){
	printf("\n%d Printing %s\n", id, name);
	int i;
	for(i=0;i<n;++i){
		printf("  %8.10f  ", vector[i]);
	}
	printf("\n");
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
