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


void DQNParallel_FBD(int N, int Dim, int size_row, int size_col, char* type, char* path){
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

	double *GradijentGlob;

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

	double *Gradijent=calloc(Dim, sizeof(double));
	double *AWeightInv=calloc(Dim*Dim, sizeof(double));
	double *ww = calloc(Dim - 1, sizeof(double));
	double *subMatr=calloc(Nx, sizeof(double));
	double vv, dot;
	double coeff, coeff2;
	double *Xdiff=calloc(Dim, sizeof(double));

	double *eye = calloc(Dim * Dim, sizeof(double));
	for (h = Dim-1; h >=0; h--)
		eye[h * Dim + h] = 1.0;

	for(int i=0;i<Dim;i++)
	   	AWeightInv[i*Dim+i] = 1.0;

	double *GradOld=calloc(Dim, sizeof(double));

	int iters=0;
	int stop=-1;

  MPI_Comm global_active_comm;
  MPI_Comm *allCurrComms;
  double oldLocalEN=0.0;
  double enLocal=0.0;
  int *active=calloc(N, sizeof(int));

  int probab_bound=0;

	for(int k=iter;k>0;--k){

		//sync point here
		MPI_Barrier(MPI_COMM_WORLD);

		//free the communicators from the previous iteration
		if(k<iter){
			if(active[my_rank])
				MPI_Comm_free(&global_active_comm);
			for(int c=0;c<N;c++){
				if(active[c]){
					if (my_rank==c || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[my_rank]))
						MPI_Comm_free(&allCurrComms[c]);
				} 
			}
			free(allCurrComms);
		}
		
		//check if someone has a stop signal, if yes, stop the algorithm
		MPI_Allreduce(&stop, &stopGlobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		if(stopGlobal>=0)
		  break;

		 probab_bound=(1/(iter-k+2))*10;
		 if(probab_bound<1)
			probab_bound=1;
		
		active=calloc(N, sizeof(int));
		int activeLoc=0;

		//calculate teh communcation probability
		srand(time(NULL)+my_rank+k);
		int comm_probab=0;
		comm_probab=rand() % 10 +1;

		int color=1;
		allCurrComms=calloc(N, sizeof(MPI_Comm));
		
		if(comm_probab > probab_bound){ //no communcation
			color=MPI_UNDEFINED;
		}
		else
			activeLoc=1;

		MPI_Allgather(&activeLoc, 1, MPI_INT, active, 1, MPI_INT, MPI_COMM_WORLD);
		MPI_Comm_split(MPI_COMM_WORLD, color, my_rank, &global_active_comm);

		for(int c=0;c<N;c++){
				if((my_rank==c && activeLoc) || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[c])){
					int my_local_rank;
					MPI_Comm_rank(allComms[c], &my_local_rank);
					MPI_Comm_split(allComms[c], color, my_local_rank, &allCurrComms[c]);
				}
		}

		
		int active_glRank;
		if(active[my_rank]){
			MPI_Comm_rank(global_active_comm, &active_glRank);
		}
		
		double *NablaPsi=calloc(Dim, sizeof(double));
		double *AWeight=calloc(Dim*Dim, sizeof(double));
		double *sDirection=calloc(Dim, sizeof(double));
		int *pivotArray=calloc(Dim, sizeof(int));
	
		compute_Gradient(N_w, N_x, N_f, rem, Dim, X, Gradient, GradOld, my_rank, Adata, Bdata, lambda_penal);

		double active_neighbours_weight=0.0;
		int my_communcator_size=0;
		double *Xremote;
		
		if(active[my_rank]){
			my_communcator_size=0;
			MPI_Comm_size(allCurrComms[my_rank], &my_communcator_size);
			Xremote=calloc((my_communcator_size)*Dim, sizeof(double));

			for(int c=0;c<N;c++){
				if(my_rank==c || (is_my_neighbour(my_rank, c, my_neighbours, my_neighbours_count) && active[c])){
					MPI_Gather(X, Dim, MPI_DOUBLE, Xremote, Dim, MPI_DOUBLE, 0, allCurrComms[c]);
				}
			}
		
			active_neighbours_weight=0.0;
			for(i=0;i<my_communcator_size-1;++i){
		    	double *zero=calloc(Dim, sizeof(double));
		    	int my_neighbours_rank=get_my_active_neighbour(i, my_rank, my_neighbours, my_neighbours_count, active);
	   	    LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X, Dim, Xdiff, Dim);
	   	    cblas_daxpy(Dim, -1, Xremote+(i+1)*Dim, 1, Xdiff, 1);

	   	    cblas_daxpy(Dim, myWMatrix[my_neighbours_rank], Xdiff, 1, zero, 1);
			    active_neighbours_weight+=myWMatrix[my_neighbours_rank];
	   	    cblas_daxpy(Dim, 1, zero, 1, NablaPsi, 1);

		    	free(zero);
	   	}
	   }
 
	   double *curRes=calloc(Dim, sizeof(double));
	   int active_node_cnt;

	   double my_self_confidence=0;
	   GradijentGlob=calloc(Dim*N, sizeof(double));

	   if(active[my_rank]){
	     MPI_Comm_size(global_active_comm, &active_node_cnt);
		   
	     my_self_confidence = 1-active_neighbours_weight;
	   	 for(i=0;i<Dim;i++){
	   			curRes[i]=(1-my_self_confidence)*X[i];
	   			for(j=1;j<=my_communcator_size-1;j++){
	   				int my_neighbours_rank=get_my_active_neighbour(j-1, my_rank, my_neighbours, my_neighbours_count, active);
	   				curRes[i]+=-myWMatrix[my_neighbours_rank]*Xremote[j*Dim+i];
	   			}
	  			curRes[i]+=stepSize*GradOld[i];
	   		}
	   	}
	   	else{
	   		my_self_confidence=1;
	   		cblas_daxpy(Dim, stepSize, GradOld, 1, curRes, 1);
	   	}
	
	   	int my_active_rank=-1;
		  if(active[my_rank])
			  MPI_Comm_rank(global_active_comm, &my_active_rank);
		
	   	MPI_Gather(curRes, Dim, MPI_DOUBLE, GradijentGlob, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

		  double euclidean_norm;
		  iters=0;
		
		  if(my_rank==0){
			  euclidean_norm=cblas_dnrm2 (Dim*N, GradijentGlob, 1);
			  if(k<iter && euclidean_norm<epsilon && euclidean_norm>0.0){
				  iters=iter-k;
			  }
			}

		  MPI_Bcast(&euclidean_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		  if(k<iter && euclidean_norm<epsilon && euclidean_norm>0.0){
			  stop=my_rank;
			  continue;
		  }
	    
      cblas_daxpy(Dim, stepSize, Gradijent, 1, NablaPsi, 1);

	    cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1.0, AWeightInv, Dim, NablaPsi, 1, 1.0, sDirection, 1);
	    cblas_daxpy(Dim, -1, sDirection, 1, X, 1);

		  free(NablaPsi);
			free(sDirection);
		  free(pivotArray);
		  if(active[my_rank]){
			  free(Xremote);
			  free(GradijentGlob);
		  }

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

	double *Xall=calloc(Dim*N, sizeof(double));
	MPI_Gather(X, Dim, MPI_DOUBLE, Xall, Dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(my_rank==0){
		FILE *file;
		char fileName[50];
		strcpy(fileName, path);
		strcat(fileName, "/");
		strcat(fileName, type);
		strcat(fileName,"log_BHOPD.txt");
		printf("%s\n", fileName);
		file=fopen(fileName,"w");
		fprintf(file, "N=%d, Dim=%d, rows=%d, cols=%d\n", N, Dim, size_row, size_col);
		fprintf(file," Total time: %f\n", max_time);
		fprintf(file," Required number of iterations: %d", iters);
		fclose(file);
	}
	free(X);
	MPI_Finalize();
}
