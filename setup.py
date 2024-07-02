from setuptools import dist, setup
from distutils.extension import Extension
dist.Distribution().fetch_build_eggs([
    'Cython>=3.0', 'numpy>=1.2'])

import numpy as np

version="0.0.11"

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

ext_modules = [
    Extension(
        name="pdb_numpy.alignement.align_cython",
        sources=["src/pdb_numpy/alignement/align_cython.pyx"],
        include_dirs=[np.get_include()]),
    Extension(
        name="pdb_numpy.format.split_cython",
        sources=["src/pdb_numpy/format/split_cython.pyx"]),
    Extension(
        name="pdb_numpy.format.encode_cython",
        sources=["src/pdb_numpy/format/encode_cython.pyx"]),
]

requirements = [
    'numpy>=1.2',
    'cython>=3.0',
]

setup(
    name='pdb_numpy',
    version=version,
    description=(
        'Pdb_Numpy is a python library allowing simple operations on pdb coor files.'
    ),
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Samuel Murail',
    author_email="samuel.murail@u-paris.fr",
    url='https://github.com/samuelmurail/pdb_numpy',
    packages=['pdb_numpy', 'pdb_numpy.data', 'pdb_numpy.format', 'pdb_numpy.alignement'],
    package_dir={'pdb_numpy': 'src/pdb_numpy'},
    entry_points={'console_scripts': ['pdb_numpy = pdb_numpy.__main__:main']},
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    license='GNUv2.0',
    zip_safe=False,
    ext_modules = ext_modules,
    package_data={
        'pdb_numpy.data': [
            'blosum62.txt',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    keywords=[
        "pdb_numpy",
        "Python",
        "PDB",
        "Numpy",
        "Coor",
        "Model",
    ],
)
