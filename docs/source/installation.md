# Installation Quick Start


## Get sources from the GithubRepo

The sources for PDB-numpy can be downloaded from the GithubRepo.

You can either clone the public repository:

```bash
$ git clone git://github.com/samuelmurail/pdb_numpy
```

Or download the tarball:

```bash
$ curl -OJL https://github.com/samuelmurail/pdb_numpy/tarball/master
```

Once you have a copy of the source, switch to the `pdb_numpy` directory.

```bash
$ cd pdb_numpy
```

##  Install `pdb_numpy`

Once you have a copy of the source and have created a conda environment, you can install it with:

```bash
$ python setup.py install
```

## Test Installation

To test the installation, simply use pytest:

```bash
$ pytest
=================== test session starts ===================
platform linux -- Python 3.8.16, pytest-7.3.0, pluggy-1.0.0
rootdir: /home/murail/Documents/Code/pdb_numpy
configfile: pytest.ini
testpaths: src/pdb_numpy/tests
plugins: anyio-3.5.0
collected 34 items                                                            

src/pdb_numpy/tests/test_DSSP.py ..                  [  5%]
src/pdb_numpy/tests/test_alignement.py .......       [ 26%]
src/pdb_numpy/tests/test_analysis.py ......          [ 44%]
src/pdb_numpy/tests/test_geom.py ..                  [ 50%]
src/pdb_numpy/tests/test_mmcif.py .....              [ 64%]
src/pdb_numpy/tests/test_pdb.py .......              [ 85%]
src/pdb_numpy/tests/test_select.py .....             [100%]

=================== 34 passed in 11.99s ===================
```
