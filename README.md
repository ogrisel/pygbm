# pygbm [![Build Status](https://travis-ci.org/ogrisel/pygbm.svg?branch=master)](https://travis-ci.org/ogrisel/pygbm)

Experimental Gradient Boosting Machines in Python.

The goal of this project is to evaluate whether it's possible to
implement a pure Python yet efficient version histogram-binning of
Gradient Boosting Trees (possibly with all the LightGBM optimizations)
while staying in pure Python using the [numba](http://numba.pydata.org/)
jit compiler.

We plan scikit-learn compatible set of estimators class and possibly
integration with dask and dask-ml for out-of-core and distributed
fitting on a cluster.

## Status

This is unusable / under development.

## Running the development version

Use pip to install in "editable" mode:

    git clone https://github.com/ogrisel/pygbm.git
    cd pygbm
    pip install --editable .

Run the tests with pytest:

    pip install -r requirements.txt
    pytest

## Benchmarking

The `benchmarks` folder contains some scripts to evaluate the computation
performance of various parts of pygbm.

### Profiling

To profile the benchmarks, you can use
[snakeviz](https://jiffyclub.github.io/snakeviz/) to get an interactive
HTML report:

    pip install snakeviz
    python -m cProfile -o bench_grower.prof benchmarks/bench_grower.py
    snakeviz bench_grower.prof

### Debugging numba type inference

To introspect the results of type inference steps in the numba sections
called by a given benchmarking script:

    numba --annotate-html bench_grower.html benchmarks/bench_grower.py

In particular it is interesting to check that the numerical variables in
the hot loops highlighted by the snakeviz profiling report have the
expected precision level (e.g. `float32` for loss computation, `uint8`
for binned feature values, ...).

### Impact of thread-based parallelism

Some benchmarks can call numba functions that leverage the built-in
thread-based parallelism with `@njit(parallel=True)` and `prange` loops.
On a multicore machine you can evaluate how the thread-based parallelism
scales by explicitly setting the `NUMBA_NUM_THREAD` environment
variable. For instance try:

    NUMBA_NUM_THREADS=1 python benchmarks/bench_binning.py

vs:

    NUMBA_NUM_THREADS=4 python benchmarks/bench_binning.py
