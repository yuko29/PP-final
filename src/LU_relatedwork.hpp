#ifndef LU_RELATED_HPP_
#define LU_RELATED_HPP_


#include "user_types.hpp"

void LUPDecompose(double **A, int N, int *P);
void LUPInvert(double **A, int *P, int N);
void LUinit(int n, double** mat_ori, double*** mat_new, int** P_new);

#endif /* LIB_TESTING_HPP_ */
