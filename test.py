import sys
sys.path.insert(0, './src')

import pdb_numpy.coor
import pdb_numpy.DSSP as DSSP

#test = pdb_numpy.Coor("./src/pdb_numpy/tests/input/1rxz.pdb")
test = pdb_numpy.Coor(pdb_id="3eam")
test.write_pdb("./tmp.pdb", check_file_out=False)

print(test.len)

print('Adding NH')
DSSP.add_NH(test)
print('NH added')

# run dssp:
# sudo apt install dssp
# dssp -i tmp.pdb -o dssp.txt

with open("./dssp.txt") as f:
    lines = f.readlines()

start_read = False
dssp_seq = ""
for line in lines:
    if line.startswith("  #  RESIDUE AA STRUCTURE BP1 BP2  ACC"):
        start_read = True
        continue
    if start_read and line[13] != "!":
        dssp_seq += line[16]

print(dssp_seq)
print(len(dssp_seq))


print('Computing DSSP')
matrix = DSSP.compute_DSSP(test)
print('DSSP computed')

import matplotlib.pyplot as plt

#plt.imshow(matrix)
#plt.show()