import sys
sys.path.insert(0, './src')

import pdb_numpy.coor
import pdb_numpy.DSSP as DSSP

test = pdb_numpy.Coor("./src/pdb_numpy/tests/input/1rxz.pdb")

print(test.len)

DSSP.add_NH(test)

test.write_pdb("./tmp.pdb", check_file_out=False)

matrix = DSSP.compute_DSSP(test)

import matplotlib.pyplot as plt

plt.imshow(matrix)
plt.show()