#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "LU_relatedwork.hpp"

/*
 *
 * Decomposition Function, the result of this function is the Matrix A decomposed in the LU form with Pivoting.
 * A= L-E+U s.t. P*A=L*U
 * This function isn't parallelized because there are a lot of critical part and using the task or the critical region
 * insert only overhead because the code must be filled with taskwait and critical region.
 */
void LUPDecompose(double **A, int N, int *P) {
    int i, j, k, imax; 
    double maxA, absA;
    int flag=1;
    for (i = 0; i <= N; i++)
        P[i] = i; 
    for (i = 0; i < N; i++) 
    {
        maxA = 0.0;
        imax = i;

        for (k = i; k < N; k++)
            if ((absA = fabs(A[k][i])) > maxA) 
            { 
          
                maxA = absA;
                imax = k;
                
            }
        if (imax != i) 
        {
            
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;
            
            //pivoting without loop (use row pointers)
            A[2*N+i] = A[i];
            A[i] = A[imax];
            A[imax] = A[2*N+i];
            P[N]++;
        
        }
        
        for (j = i + 1; j < N; j++) 
        {
            A[j][i] /= A[i][i];

            for (k = i + 1; k < N; k++)
                A[j][k] -= A[j][i] * A[i][k];
        }
    }
}
/*
*
* This function compute the inverse, simpler is faster, parallelize the outer loop to have better performance
* i have tried to making a collapse but the results is the same.
*
**/
void LUPInvert(double **A, int *P, int N) {
    int chunk = 10;
  	#pragma omp parallel shared(A, P, N, chunk)
  	{
        #pragma omp for schedule(dynamic, chunk)
        for (int j = 0; j < N; j++) {
            
            for (int i = 0; i < N; i++) {
                if (P[i] == j) 
                    A[N+i][j] = 1.0;
                else
                    A[N+i][j] = 0.0;
                for (int k = 0; k < i; k++)
                    A[N+i][j] -= A[i][k] * A[N+k][j]; //Computation of the result matrix
            }
            for (int i = N - 1; i >= 0; i--) {
                for (int k = i + 1; k < N; k++)
                    A[N+i][j] -= A[i][k] * A[N+k][j];
                A[N+i][j] = A[N+i][j] / A[i][i];
            }
        }
    }
}

void LUinit(int n, double** mat_ori, double*** mat_new, int** P_new)
{
	double **A=(double **)malloc(3*n*sizeof(double *)); // in this version of the code the matrix the three matrix are contiguous but 
	//i have used a different type of pointer to avoid the loop of pivoting. i need to add a loop to make a pointer of rows.
	for (int i = 0; i < 2*n; i++) 
  		A[i] = (double *)malloc(n*sizeof(double));

	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			A[i][j] = mat_ori[i][j];

	int *P=(int *)malloc((n+1)*sizeof(int));
    *P_new = P;
    *mat_new = A;
}