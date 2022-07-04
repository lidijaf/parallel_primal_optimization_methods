#include <lapacke_utils.h>

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
