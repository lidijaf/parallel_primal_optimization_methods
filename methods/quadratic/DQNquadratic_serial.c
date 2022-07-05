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
#include <sysexits.h>
#include <err.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

void DQNquadraticSimpl(double *Adata, double *Bdata, int *A, int *degreeSensor, int N, int Dim);
double max(double* a, int Dim);
void printVector(double* vector, int n, char* name);

int main(int argc, char* argv[]) {

	if (argc < 3) {
		printf("Input parameters N and Dim not specified.");
		return -1;
	}
	int N = atoi(argv[1]);
	int Dim = atoi(argv[2]);

	clock_t begin = clock();

	double *Adata = calloc(N * Dim * Dim, sizeof(double));
	double *Bdata = calloc(N * Dim, sizeof(double));
	int *Adj = calloc(N * N, sizeof(double));
	int *degreeSensor = calloc(N, sizeof(double));
	int fd;
	char *infile = "Adata.bin";
	int bytes_expected = N * Dim * Dim * sizeof(double);
	fd = open(infile, O_RDONLY);
	read(fd, Adata, bytes_expected);

	infile = "Bdata.bin";
	bytes_expected = N * Dim * Dim * sizeof(double);
	fd = open(infile, O_RDONLY);
	read(fd, Bdata, bytes_expected);

	infile = "Adj.bin";
	fd = open(infile, O_RDONLY);
	read(fd, Adj, bytes_expected);

	infile = "degSens.bin";
	bytes_expected = N * sizeof(int);
	fd = open(infile, O_RDONLY);
	read(fd, degreeSensor, bytes_expected);

	DQNquadraticSimpl(Adata, Bdata, Adj, degreeSensor, N, Dim);
	clock_t end = clock();
	double time_spent = (double) (end - begin) / CLOCKS_PER_SEC;
	printf("\nElapsed time: %f", time_spent);
	return 0;
}

void DQNquadraticSimpl(double *Adata, double *Bdata, int *A, int *degreeSensor, int N, int Dim) {
	double *LTemp = calloc(N, sizeof(double));
	int i;
	int j;

	double *AdataCopy = calloc(N * Dim * Dim, sizeof(double));
	LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', N * Dim, Dim, Adata, Dim, AdataCopy, Dim);
	for (i = N-1; i >= 0; --i) {
		double *wr = calloc(Dim, sizeof(double)), *wi = calloc(Dim, sizeof(double)), *vl = calloc(Dim * Dim, sizeof(double)), *vr = calloc(Dim * Dim, sizeof(double));
		double *subMatr = &(AdataCopy[(i * Dim) * Dim]);
		LAPACKE_dgeev( LAPACK_ROW_MAJOR, 'N', 'N', Dim, subMatr, Dim, wr, wi, vl, Dim, vr, Dim);
		LTemp[i] = max(wr, Dim);
		free(wr);
		free(wi);
		free(vl);
		free(vr);
	}
	double LConst = max(LTemp, Dim);
	free(LTemp);
	double stepSize = 1 / LConst / 200;
	double *WMatrix = calloc(N * N, sizeof(double));
	for (i = N - 2; i >= 0; --i) {
		for (j = N - 1; j >= i + 1; --j) {
			if (A[i * N + j]) {
				WMatrix[i * N + j] = 1.0 / (1.0 + MAX(degreeSensor[i], degreeSensor[j]));
				WMatrix[j * N + i] = WMatrix[i * N + j];
			}
		}
	}
	double *eye = calloc(N * N, sizeof(double));
	double ones[N];
	for (i = N - 1; i >= 0; --i) {
		eye[i * N + i] = 1.0;
		ones[i] = 1.0;
	}
	double *v = calloc(N, sizeof(double));
	cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1, WMatrix, N, ones, 1, 0, v, 1);
	double *diagMatr = calloc(N * N, sizeof(double));
	for (i = N - 1; i >= 0; --i)
		diagMatr[i * N + i] = v[i];
	cblas_daxpy(N * N, -1, diagMatr, 1, eye, 1);
	cblas_daxpy(N * N, 1, eye, 1, WMatrix, 1);
	double *eyeScaled = calloc(N * N, sizeof(double));
	for (i = N - 1; i >= 0; --i) {
		eyeScaled[i * N + i] = 0.5;
	}
	for (i = N * N - 1; i >= 0; --i)
		WMatrix[i] *= 0.5;
	cblas_daxpy(N * N, 1, eyeScaled, 1, WMatrix, 1);
	free(eye);
	free(eyeScaled);
	free(diagMatr);
	free(v);

	double *X = calloc(N * Dim, sizeof(double));
	double *Gradijent = calloc(N * Dim, sizeof(double));
	int k;
	for (k = 2000; k > 0; --k) {
		double *NablaPsi = calloc(Dim*N, sizeof(double));
		double *AWeight = calloc(Dim*N*N * Dim, sizeof(double));
		double *AWeightCpy = calloc(Dim*N*N * Dim, sizeof(double));
		double *sDirection = calloc(N*Dim, sizeof(double));
		double *GMatrix = calloc(Dim * N * N * Dim, sizeof(double));
		double *BdataCopy = calloc(N*Dim, sizeof(double));
		double *copyX = calloc(Dim*N, sizeof(double));
		int *pivotArray = calloc(Dim*N, sizeof(int));
		double *WKron = calloc(N * Dim * N * Dim, sizeof(double));

		for (i = N-1; i >= 0; --i) {
			LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, X + i * Dim, Dim, copyX, Dim);
			LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', 1, Dim, Bdata + i * Dim, Dim, BdataCopy, Dim);
			cblas_daxpy(Dim, -1, BdataCopy, 1, copyX, 1);
			cblas_dgemv(CblasRowMajor, CblasNoTrans, Dim, Dim, 1, Adata + i * Dim * Dim, Dim, copyX, 1, 0, Gradijent + i * Dim, 1);
		}
		for (i = N-1; i >= 0; --i) {
			LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, Adata + i * Dim * Dim, Dim, GMatrix + i * Dim * N * Dim + i * Dim, Dim * N);
		}
		double *eyeDim = calloc(Dim * Dim, sizeof(double));
		for (i = Dim-1; i >= 0; --i)
			eyeDim[i * Dim + i] = 1.0;
		for (i = N*N-1; i >= 0; --i) {
			double *y = calloc(Dim * Dim, sizeof(double));
			cblas_daxpy(Dim * Dim, WMatrix[i], eyeDim, 1, y, 1);
			LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', Dim, Dim, y, Dim, WKron + (i / N) * Dim * Dim * N + Dim * (i % N), Dim * N);
		}
		double *WDiag=calloc(N * Dim * N * Dim, sizeof(double));
		for (i = N*Dim-1; i  >= 0; --i) {
			WDiag[i * N * Dim + i] = WKron[i * N * Dim + i];
		}
		double *Wu=calloc(N * Dim * N * Dim, sizeof(double));
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', N * Dim, N * Dim, WKron, N * Dim, Wu, N * Dim);
		cblas_daxpy(N * Dim * N * Dim, -1, WDiag, 1, Wu, 1);
		double *NablaDvaF=calloc(N * Dim * N * Dim, sizeof(double));
		LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', N * Dim, N * Dim, GMatrix, N * Dim, NablaDvaF, N * Dim);
		double *eyeNDim=calloc(N * Dim * N * Dim, sizeof(double));
		for (i = N*Dim-1; i >= 0; --i)
			eyeNDim[i * N * Dim + i] = 1.0;
		double *WKronCpy=calloc(N * Dim * N * Dim, sizeof(double));
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', N * Dim, N * Dim, WKron, N * Dim, WKronCpy, N * Dim);
		cblas_daxpy(N * Dim * N * Dim, -1, WKronCpy, 1, eyeNDim, 1);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N * Dim, N * Dim, 1, eyeNDim, N * Dim, X, 1, 0, NablaPsi, 1);
		cblas_daxpy(N * Dim, stepSize, Gradijent, 1, NablaPsi, 1);
		for (i = N*Dim-1; i >= 0; --i)
			AWeight[i * N * Dim + i] = 1.0;
		cblas_daxpy(N * Dim * N * Dim, stepSize, NablaDvaF, 1, AWeight, 1);
		cblas_daxpy(N * Dim * N * Dim, -1, WDiag, 1, AWeight, 1);
		LAPACKE_dlacpy( LAPACK_ROW_MAJOR, 'A', N * Dim, N * Dim, AWeight, N * Dim, AWeightCpy, N * Dim);
		LAPACKE_dgetrf(LAPACK_ROW_MAJOR, N * Dim, N * Dim, AWeightCpy, N * Dim, pivotArray);
		LAPACKE_dgetri( LAPACK_ROW_MAJOR, N * Dim, AWeightCpy, N * Dim, pivotArray);
		cblas_dgemv(CblasRowMajor, CblasNoTrans, N * Dim, N * Dim, 1.0, AWeightCpy, N * Dim, NablaPsi, 1, 1.0, sDirection, 1);
		cblas_daxpy(N * Dim, -1, sDirection, 1, X, 1);

		free(NablaPsi);
		free(AWeight);
		free(AWeightCpy);
		free(sDirection);
		free(GMatrix);
		free(BdataCopy);
		free(copyX);
		free(pivotArray);
		free(WKron);
		free(Wu);
		free(NablaDvaF);
	}
	free(Gradijent);
	printVector(X, N * Dim, "X-final solution");
	free(X);

}

void printVector(double* vector, int n, char* name) {
	printf("\nPrinting %s\n", name);
	int i;
	for (i = 0; i < n; i++)
		printf("%20.14f  ", vector[i]);
	printf("\n");
}

double max(double* a, int Dim) {
	double maxVal = a[0];
	int j;
	for (j = 1; j < Dim; j++)
		maxVal = MAX(maxVal, a[j]);
	return maxVal;
}
