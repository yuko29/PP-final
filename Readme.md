# Parallelization of Matrix Inversion

Parallel Programming 2022 final project.

We do the parallel optimization to the LU-decomposition based matrix inversion method by OpenMP.
The original serial program is from [gauss-benchmark-eigen](https://github.com/mndxpnsn/gauss-benchmark-eigen)

We also integrated the similiar work [ParallelMatrix](https://github.com/OcraM17/ParallelMatrix) to evalutate the performace of our proposed implementation.

## Build

```
make
```

## Usage

```
$ ./main --help
Usage: ./main [options]
Program Options:
The <UINT> below should > 0.
  -n <UINT>    Set matrix dimension (500 * 500 is default).
  -s <UINT>    Set the seed of matrix generation.
  -b <INT>     Set the min(begin) value of matrix's elements (-25 is default).
  -e <INT>     Set the max(end) value of matrix's elements (25 is default).
  -t <UINT>    Set the number of threads to run. (if availabe, 4 is default.)
  -r <UINT>    Set the repeat times of running PP to get average time (1 time is default).
  -R           Only run relatedwork. (run relatedwork and PP is default)
  -P           Only run PP. (run relatedwork and PP is default)
  -v           Verify PP's answer with serial's.
  -h           This message.
```
