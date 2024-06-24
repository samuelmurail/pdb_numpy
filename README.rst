
.. image:: https://readthedocs.org/projects/pdb-numpy/badge/?version=latest
    :target: https://pdb-numpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://dev.azure.com/samuelmurailRPBS/pdb_numpy/_apis/build/status%2Fsamuelmurail.pdb_numpy?branchName=main
    :target: https://dev.azure.com/samuelmurailRPBS/pdb_numpy/_build/latest?definitionId=1&branchName=main
    :alt: Build Status

.. image:: https://codecov.io/gh/samuelmurail/pdb_numpy/branch/main/graph/badge.svg?token=MCVDZ7GD0V
    :target: https://codecov.io/gh/samuelmurail/pdb_numpy
    :alt: Code coverage

.. image:: https://img.shields.io/pypi/v/pdb-numpy
    :target: https://pypi.org/project/pdb-numpy/
    :alt: PyPI - Version

.. image:: https://static.pepy.tech/badge/pdb-numpy
    :target: https://pepy.tech/project/pdb-numpy
    :alt: Downloads


About PDB-numpy
===============

.. image:: https://raw.githubusercontent.com/samuelmurail/pdb_numpy/master/docs/source/logo.jpeg
  :width: 200
  :align: center
  :alt: PDB Numpy Logo


``pdb_numpy`` is a python library designed to facilitate working with PDB files
in the context of structural bioinformatics. The library builds upon the
powerful ``numpy`` library to provide efficient and easy-to-use tools for
reading, manipulating, and analyzing PDB files.

The library includes a number of functions for working with PDB files,
including functions for parsing PDB files and extracting relevant information,
such as atomic coordinates, residue identities, and structural information.
Additionally, ``pdb_numpy`` provides a range of functions for performing common
manipulations on PDB structures, such as aligning structures, superimposing
structures, and calculating RMSD values.


* Source code repository:
   https://github.com/samuelmurail/pdb_numpy


Main features:
--------------

- Reading and writing PDB/MMCIF files
- Selecting atoms
- Superimposing structures using sequences alignment
- RMSD calculation
- DockQ calculation
- Secondary Structure calculation (pseudo DSSP)

For more examples and documentation, see the ``pdb_numpy`` documentation at
https://pdb-numpy.readthedocs.io/en/latest/readme.html.

Installation
------------

``pdb_numpy`` is available on PyPI and can be installed using ``pip``:

.. code-block:: bash

    pip install pdb_numpy

Alternatively, you can install ``pdb_numpy`` from source:

.. code-block:: bash

    git clone https://github.com/samuelmurail/pdb_numpy
    cd pdb_numpy
    python setup.py install

Dependencies
------------

``pdb_numpy`` requires the following dependencies:

- ``numpy``
- ``cython``


Contributing
------------

``pdb_numpy`` is an open-source project and contributions are welcome. If
you find a bug or have a feature request, please open an issue on the GitHub
repository at https://github.com/samuelmurail/pdb_numpy. If you would like
to contribute code, please fork the repository and submit a pull request.

To build locally the extension module, you can run the following command:

```bash
python setup.py build_ext --inplace
```

Author
--------------

* `Samuel Murail <https://samuelmurail.github.io/PersonalPage/>`_, Associate Professor - `Université Paris Cité <https://u-paris.fr>`_, `CMPLI <http://bfa.univ-paris-diderot.fr/equipe-8/>`_.

See also the list of `contributors <https://github.com/samuelmurail/pdb_numpy/contributors>`_ who participated in this project.

License
--------------

This project is licensed under the GNU General Public License v2.0 - see the ``LICENSE`` file for details.
