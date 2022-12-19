//
//  main.cpp
//  gauss-jordan
//
//  Created by mndx on 17/04/2022.
//

#include <chrono>

// #include "Eigen/LU"
// #include "lib_gauss.hpp"
#include "lib_mat.hpp"
#include "lib_mem.hpp"
#include "lib_testing.hpp"
#include "lib_testing_ref.hpp"
#include "user_types.hpp"
#include "parallel_inv.hpp"

// add by myself
#include <unistd.h>
//

using namespace std;
using namespace std::chrono;

// add by myself
#define TOLERANCE 1e-5

void usage(const char *progname)
{
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("The <UINT> below should > 0.\n");
    printf("  -n <UINT>     Set matrix dimension (500 * 500 is default).\n");
    printf("  -s <UINT>     Set the seed of matrix generation.\n");
    printf("  -b <INT>     Set the min(begin) value of matrix's elements (-25 is default).\n");
    printf("  -e <INT>     Set the max(end) value of matrix's elements (25 is default).\n");
    printf("  -t <UINT>     Set the times of running PP to get average time (1 time is default).\n");
    printf("  -v           Verify PP's answer with serial's.\n");
    printf("  -h           This message.\n");
    exit(0);
}

void verifyResult(const i_real_matrix &matPP, const i_real_matrix &matAns, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (abs(matPP[i][j] - matAns[i][j]) > TOLERANCE)
            {
                printf("Mismatch : [%d][%d], Expected : %lf, Actual : %lf\n", i, j, matPP[i][j], matAns[i][j]);
                return;
            }
        }
    }
}
//

int main(int argc, char * argv[]) {

    // Size input matrix
    int n = 500;

    // add by myself
    int times = 1, beg_val = -25, end_val = 25;
    bool verify = false;
    unsigned int seed = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:s:b:e:t:vh")) != -1)
    {
        switch (opt)
        {
            case 'n':
                n = atoi(optarg);
                if (n <= 0) usage(argv[0]);
                break;
            case 's':
                seed = atoi(optarg);
                if (seed <= 0) usage(argv[0]);
                break;
            case 'b':
                beg_val = atoi(optarg);
                break;
            case 'e':
                end_val = atoi(optarg);
                break;
            case 't':
                times = atoi(optarg);
                if (times <= 0) usage(argv[0]);
                break;
            case 'v':
                verify = true;
                break;
            case 'h':
                usage(argv[0]);
            default:
                usage(argv[0]);
        }
    }
    //


    // Allocate space for matrices
    double ** mat_ori = mat2D(n);
    i_real_matrix mat;

    // Populate matrix mat with some data
    init_mat(n, mat_ori, seed, beg_val, end_val);

    // Populate reference matrix mat with mat data
    set_mat_to_vec2D(mat_ori, n, mat);

    // Free allocated space
    free_mat2D(mat_ori, n);

    // set time data type
    std::chrono::time_point<std::chrono::high_resolution_clock>  start, stop;
    milliseconds duration;

    // add by myself
    i_real_matrix mat_inv;
    if (verify)
    {
    //
        // Time serial
        start = high_resolution_clock::now();

        // Compute inverse using serial
        mat_inv = inv_ref(mat, true);

        // Get stop time serial
        stop = high_resolution_clock::now();

        // Get duration serial
        duration = duration_cast<milliseconds>(stop - start);

        // Print duration of serial
        cout << "duration serial: " << duration.count() << " (ms)\n";
    // add by myself
    }
    //
    cout << "duration openmp: \n";

    i_real_matrix mat_inv_PP;
    vector<milliseconds> durations(times);
    for (int i = 0; i < times; ++i)
    {
        // Time PP
        start = high_resolution_clock::now();

        // Compute inverse using PP
        mat_inv_PP = inv_ref_PP(mat, true);

        // Get stop time PP
        stop = high_resolution_clock::now();

        // Get duration PP
        durations[i] = duration_cast<milliseconds>(stop - start);

        // Print duration of PP
        cout << "number" << i << ": " << durations[i].count() << " (ms)\n";        
    }

    sort(durations.begin(), durations.end());
    milliseconds sum_durations = duration_cast<milliseconds>(start - start); // set to zero
    // comput average, but the biggst and smallest 10% time will not be computed
    int starti = (float)times * 0.1, endi = (float)times * 0.9 + 1;
    for (int i = starti; i < endi; ++i)
        sum_durations += durations[i];
    sum_durations /= endi - starti;
    cout << "average duration openmp: " << sum_durations.count() << " (ms)\n";      

    // add by myself
    if (verify) verifyResult(mat_inv_PP, mat_inv, mat_inv.size());
    //

    return 0;
}



