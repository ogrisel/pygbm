# pygbm [![Build Status](https://travis-ci.org/ogrisel/pygbm.svg?branch=master)](https://travis-ci.org/ogrisel/pygbm) [![codecov](https://codecov.io/gh/ogrisel/pygbm/branch/master/graph/badge.svg)](https://codecov.io/gh/ogrisel/pygbm) [![python versions](https://img.shields.io/badge/python-3.6+-blue.svg)](https://github.com/ogrisel/pygbm)



Experimental Gradient Boosting Machines in Python.

The goal of this project is to evaluate whether it's possible to implement a
pure Python yet efficient version histogram-binning of Gradient Boosting
Trees (possibly with all the LightGBM optimizations) while staying in pure
Python 3.6+ using the [numba](http://numba.pydata.org/) jit compiler.

pygbm provides a set of scikit-learn compatible estimator classes that
should play well with the scikit-learn `Pipeline` and model selection tools
(grid search and randomized hyperparameter search).

Longer term plans include integration with dask and dask-ml for
out-of-core and distributed fitting on a cluster.

## Installation

The project is available on PyPI and can be installed with `pip`:

    pip install pygbm

You'll need Python 3.6 at least.

## Documentation

The API documentation is available at:

https://pygbm.readthedocs.io/

You might also want to have a look at the `examples/` folder of this repo.

## Status

The project is experimental. The API is subject to change without deprecation notice. Use at your own risk.

We welcome any feedback in the github issue tracker:

https://github.com/ogrisel/pygbm/issues

## Running the development version

Use pip to install in "editable" mode:

    git clone https://github.com/ogrisel/pygbm.git
    cd pygbm
    pip install -r requirements.txt
    pip install --editable .

Run the tests with pytest:

    pip install -r requirements.txt
    pytest

## Benchmarking

The `benchmarks` folder contains some scripts to evaluate the computation
performance of various parts of pygbm. Keep in mind that numba's JIT
compilation [takes
time](http://numba.pydata.org/numba-doc/latest/user/5minguide.html#how-to-measure-the-performance-of-numba)!

### Profiling

To profile the benchmarks, you can use
[snakeviz](https://jiffyclub.github.io/snakeviz/) to get an interactive
HTML report:

    pip install snakeviz
    python -m cProfile -o bench_higgs_boson.prof benchmarks/bench_higgs_boson.py
    snakeviz bench_higgs_boson.prof

### Debugging numba type inference

To introspect the results of type inference steps in the numba sections
called by a given benchmarking script:

    numba --annotate-html bench_higgs_boson.html benchmarks/bench_higgs_boson.py

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


## Acknowledgements

The work from Nicolas Hug is supported by the National Science Foundation
under Grant No. 1740305 and by DARPA under Grant No. DARPA-BAA-16-51

The work from Olivier Grisel is supported by the [scikit-learn initiative
and its partners at Inria Fondation](https://scikit-learn.fondation-inria.fr/en/)
