#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#include "parallel_inv.hpp"

using namespace std;

// Generate and fill an nrows x ncols matrix, fill with given value (zero by default)
i_real_matrix initRealMatrixPP(const std::size_t nrows, const std::size_t ncols, const i_float_t initValue = 0.0)
{
    return i_real_matrix(nrows, i_real_vector(ncols, initValue));
}

i_real_matrix inv_ref_PP(const i_real_matrix &matG, const bool usePermute = true)
{
    const std::size_t nrows{matG.size()}, ncols{matG[0].size()};
    const std::size_t nSize{nrows};
    i_real_matrix matLU(nSize);
    if (nrows != ncols)
    {
        std::cout << "Error when using inv: matrix is not square.\n";
        return matLU;
    }

    std::size_t i{0}, j{0}, k{0};

    // ******************** Step 1: row permutation (swap diagonal zeros) ********************
    /*std::vector<std::size_t> permuteLU(nSize); // Permute vector
    for (i = 0; i < nSize; ++i)
    {
        permuteLU.push_back(i); // Push back row index
    }*/
    std::vector<std::size_t> permuteLU(nSize); // Permute vector
    #pragma omp parallel for schedule(static, 1)
    for (i = 0; i < nSize; ++i)
        permuteLU[i] = i; // Push back row index

    if (usePermute) // Sort rows by pivot element
    {
        for (j = 0; j < nSize; ++j)
        {
            i_float_t maxv{0.0};
            for (i = j; i < nSize; ++i)
            {
                const i_float_t currentv{std::abs(matG[permuteLU[i]][j])};
                if (currentv > maxv) // Swap rows
                {
                    maxv = currentv;
                    const std::size_t tmp{permuteLU[j]};
                    permuteLU[j] = permuteLU[i];
                    permuteLU[i] = tmp;
                }
            }
        }
        /*for (i = 0; i < nSize; ++i)
        {
            matLU.push_back(matG[permuteLU[i]]); // Make a permuted matrix with new row order
        }*/
        #pragma omp parallel for schedule(static, 1)
        for (i = 0; i < nSize; ++i)
            matLU[i] = matG[permuteLU[i]]; // Make a permuted matrix with new row order
    }
    else
    {
        matLU = i_real_matrix(matG); // Simply duplicate matrix
    }

    // ******************** Step 2: LU decomposition (save both L & U in matLU) ********************
    if (matLU[0][0] == 0.0)
    {
        std::cout << "Warning when using inv: matrix is singular.\n";
        matLU.clear();
        return matLU;
    }
    #pragma omp parallel for
    for (i = 1; i < nSize; ++i)
        matLU[i][0] /= matLU[0][0]; // Initialize first column of L matrix
    #pragma omp parallel for schedule(static, 1)
    for (i = 1; i < nSize; ++i)
    {
        for (j = i; j < nSize; ++j)
        {
            for (k = 0; k < i; ++k)
            {
                matLU[i][j] -= matLU[i][k] * matLU[k][j]; // Calculate U matrix
            }
        }
        /*if (matLU[i][i] == 0.0)
        {
            std::cout << "Warning when using inv: matrix is singular.\n";
            matLU.clear();
            return matLU;
        }*/
        for (k = i + 1; k < nSize; ++k)
        {
            for (j = 0; j < i; ++j)
            {
                matLU[k][i] -= matLU[k][j] * matLU[j][i]; // Calculate L matrix
            }
            matLU[k][i] /= matLU[i][i];
        }
    }

    // ******************** Step 3: L & U inversion (save both L^-1 & U^-1 in matLU_inv) ********************
    i_real_matrix matLU_inv = initRealMatrixPP(nSize, nSize);

    // matL inverse & matU inverse
    #pragma omp parallel for schedule(static, 1)
    for (i = 0; i < nSize; ++i)
    {
        // L matrix inverse, omit diagonal ones
        matLU_inv[i][i] = 1.0;
        for (k = i + 1; k < nSize; ++k)
        {
            for (j = i; j <= k - 1; ++j)
            {
                matLU_inv[k][i] -= matLU[k][j] * matLU_inv[j][i];
            }
        }
        // U matrix inverse
        matLU_inv[i][i] = 1.0 / matLU[i][i];
        for (k = i; k > 0; --k)
        {
            for (j = k; j <= i; ++j)
            {
                matLU_inv[k - 1][i] -= matLU[k - 1][j] * matLU_inv[j][i];
            }
            matLU_inv[k - 1][i] /= matLU[k - 1][k - 1];
        }
    }

    // ******************** Step 4: Calculate G^-1 = U^-1 * L^-1 ********************
    // Lower part product
    #pragma omp parallel for schedule(static, 1)
    for (i = 1; i < nSize; ++i)
    {
        for (j = 0; j < i; ++j)
        {
            const std::size_t jp{permuteLU[j]}; // Permute column back
            matLU[i][jp] = 0.0;
            for (k = i; k < nSize; ++k)
            {
                matLU[i][jp] += matLU_inv[i][k] * matLU_inv[k][j];
            }
        }
    }
    // Upper part product
    #pragma omp parallel for schedule(static, 1)
    for (i = 0; i < nSize; ++i)
    {
        for (j = i; j < nSize; ++j)
        {
            const std::size_t jp{permuteLU[j]}; // Permute column back
            matLU[i][jp] = matLU_inv[i][j];
            for (k = j + 1; k < nSize; ++k)
            {
                matLU[i][jp] += matLU_inv[i][k] * matLU_inv[k][j];
            }
        }
    }
    return matLU; // Reused matLU as a result container
}

