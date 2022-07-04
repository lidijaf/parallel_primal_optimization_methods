#ifndef UTILITY_H
#define UTILITY_H

#include <lapacke_utils.h>


double min(double* a, int Dim);

int minInt(int* a, int Dim);

double max(double* a, int Dim);

void find_optimal_distribution(int *N_f, int *rem, int size_row, int N);

#endif
