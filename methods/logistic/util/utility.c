#include "utility.h"

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
