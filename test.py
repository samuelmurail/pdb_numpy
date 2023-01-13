import sys
sys.path.insert(0, '/home/murail/Documents/Code/pdb_numpy/src')

import pdb_numpy.coor

test = pdb_numpy.coor.Coor('/home/murail/Documents/Code/pdb_numpy/src/pdb_numpy/tests/input/1y0m.pdb')

print(test.len)