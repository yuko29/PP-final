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
#include "LU_relatedwork.hpp"
#include <unistd.h>
#include <omp.h>
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
    printf("  -n <UINT>    Set matrix dimension (500 * 500 is default).\n");
    printf("  -s <UINT>    Set the seed of matrix generation.\n");
    printf("  -b <INT>     Set the min(begin) value of matrix's elements (-25 is default).\n");
    printf("  -e <INT>     Set the max(end) value of matrix's elements (25 is default).\n");
    printf("  -t <UINT>    Set the number of threads to run. (if availabe, 4 is default.)\n");
    printf("  -r <UINT>    Set the repeat times of running PP to get average time (1 time is default).\n");
    printf("  -R           Only run relatedwork. (run relatedwork and PP is default)\n");
    printf("  -P           Only run PP. (run relatedwork and PP is default)\n");
    printf("  -v           Verify PP's answer with serial's.\n");
    printf("  -h           This message.\n");
    exit(0);
}

bool verifyResult(const i_real_matrix &matPP, const i_real_matrix &matAns, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (abs(matPP[i][j] - matAns[i][j]) > TOLERANCE)
            {
                printf("Mismatch : [%d][%d], Expected : %lf, Actual : %lf\n", i, j, matPP[i][j], matAns[i][j]);
                return false;
            }
        }
    }
    return true;
}

bool verifyResult_2(double ** matPP, double ** matAns, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (abs(matPP[i][j] - matAns[i][j]) > TOLERANCE)
            {
                printf("Mismatch : [%d][%d], Expected : %lf, Actual : %lf\n", i, j, matPP[i][j], matAns[i][j]);
                return false;
            }
        }
    }
    return true;
}
//

int main(int argc, char * argv[]) {

    // Size input matrix
    int n = 500;

    // add by myself
    int repeat = 1, beg_val = -25, end_val = 25, threads = 4;
    bool verify = false, runPP = true, runRelated = true;
    unsigned int seed = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:s:b:e:t:r:vRPh")) != -1)
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
                threads = atoi(optarg);
                if (threads <= 0) usage(argv[0]);
                break;
            case 'r':
                repeat = atoi(optarg);
                if (repeat <= 0) usage(argv[0]);
                break;
            case 'R':
                runPP = false;
                break;
            case 'P':
                runRelated = false;
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

    if (beg_val >= end_val)
    {
        cout << "min value should < max value.\n";
        return 0;
    }
    //


    // Allocate space for matrices
    double ** mat_ori = mat2D(n);
    i_real_matrix mat;

    // Populate matrix mat with some data
    init_mat(n, mat_ori, seed, beg_val, end_val);

    // Populate reference matrix mat with mat data
    set_mat_to_vec2D(mat_ori, n, mat);

    // set time data type
    std::chrono::time_point<std::chrono::high_resolution_clock>  start, stop;
    milliseconds duration;

    i_real_matrix mat_inv;
    if (verify)
    {
        // Time serial
        start = high_resolution_clock::now();

        // Compute inverse using serial
        mat_inv = inv_ref(mat, true);

        // Get stop time serial
        stop = high_resolution_clock::now();

        // Get duration serial
        duration = duration_cast<milliseconds>(stop - start);

        // Print duration of serial
        printf("----------------------------------------------------------------------------------------\n");
        cout << "duration serial: " << duration.count() << " (ms)\n";
    }

    // set threads
    printf("----------------------------------------------------------------------------------------\n");
    printf("Max system threads = %d\n", omp_get_max_threads());
    threads = std::min(threads, omp_get_max_threads());
    omp_set_num_threads(threads);
    printf("Running with %d threads\n", threads);

    if (runRelated)
    {
        // relatedwork version
        printf("----------------------------------------------------------------------------------------\n");
        cout << "duration relatedwork: \n";
        // init
        /*double** mat_related;
        int *permute_related;
        LUinit(n, mat_ori, &mat_related, &permute_related);*/

        double** mat_related = (double **)malloc(3*n*sizeof(double *));;
        int *permute_related = (int *)malloc((n+1)*sizeof(int));
        for (int i = 0; i < 2*n; i++) 
  		    mat_related[i] = (double *)malloc(n*sizeof(double));

        vector<milliseconds> durations(repeat);
        for (int i = 0; i < repeat; ++i)
        {
            for(int i=0;i<n;i++)
                for(int j=0;j<n;j++)
                    mat_related[i][j] = mat_ori[i][j];

            //if (verifyResult_2(mat_ori, mat_related, n)) 
            //    cout << "related init sucess.\n";
            
            // Time related
            start = high_resolution_clock::now();

            // decompose
            LUPDecompose(mat_related, n, permute_related);

            // inverse (answers are in mat_related)
            LUPInvert(mat_related, permute_related, n);
            
            // Get stop time related
            stop = high_resolution_clock::now();

            // Get duration related
            //duration = duration_cast<milliseconds>(stop - start);
            durations[i] = duration_cast<milliseconds>(stop - start);

            // Print duration of related
            cout << "number" << i << ": " << durations[i].count() << " (ms)\n";  
        }

        // comput average, but the biggst and smallest 10% time will not be computed
        sort(durations.begin(), durations.end());
        milliseconds sum_durations = duration_cast<milliseconds>(start - start); // set to zero
        int starti = (float)repeat * 0.1, endi = (float)repeat * 0.9 + 1;
        for (int i = starti; i < endi; ++i)
            sum_durations += durations[i];
        sum_durations /= endi - starti;
        printf("------------------------------------------------\n");
        cout << "average duration related: " << sum_durations.count() << " (ms)\n";

        if (verify)
        {
            // ans is in mat_related, but it is double**
            i_real_matrix mat_related_inv;
            set_mat_to_vec2D(mat_related + n * 1, n, mat_related_inv);
            if (verifyResult(mat_related_inv, mat_inv, n))
            {
                printf("------------------------------------------------\n");   
                cout << "Relatedwork's answer is Correct.\n";
            }
        }
    }

    if (runPP)
    {
        // PP openmp
        printf("----------------------------------------------------------------------------------------\n");
        cout << "duration openmp: \n";

        i_real_matrix mat_inv_PP;
        vector<milliseconds> durations(repeat);
        for (int i = 0; i < repeat; ++i)
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

        // comput average, but the biggst and smallest 10% time will not be computed
        sort(durations.begin(), durations.end());
        milliseconds sum_durations = duration_cast<milliseconds>(start - start); // set to zero
        int starti = (float)repeat * 0.1, endi = (float)repeat * 0.9 + 1;
        for (int i = starti; i < endi; ++i)
            sum_durations += durations[i];
        sum_durations /= endi - starti;
        printf("------------------------------------------------\n");
        cout << "average duration openmp: " << sum_durations.count() << " (ms)\n";   

        if (verify && verifyResult(mat_inv_PP, mat_inv, mat_inv.size()))
        {
            printf("------------------------------------------------\n");   
            cout << "PP's answer is Correct.\n";
        }
    }
      

    // Free allocated space
    free_mat2D(mat_ori, n);

    return 0;
}



