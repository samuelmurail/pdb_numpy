from setuptools import setup, find_packages

version="0.0.2"

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = [
    'numpy>=1.2',
    'scipy>=1.5',
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
    packages=['pdb_numpy', 'pdb_numpy.data', 'pdb_numpy.format'],
    package_dir={'pdb_numpy': 'src/pdb_numpy'},
    entry_points={'console_scripts': ['pdb_numpy = pdb_numpy.__main__:main']},
    include_package_data=True,
    python_requires='>=3.7',
    install_requires=requirements,
    license='GNUv2.0',
    zip_safe=False,
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
