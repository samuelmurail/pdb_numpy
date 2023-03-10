import sys
import time
sys.path.insert(0, './src')

import pdb_numpy.coor
import pdb_numpy.DSSP as DSSP

#test = pdb_numpy.Coor("./src/pdb_numpy/tests/input/1rxz.pdb")
test = pdb_numpy.Coor(pdb_id="3eam")
test.write_pdb("./tmp.pdb", check_file_out=False)

# run dssp:
# sudo apt install dssp
# time dssp -i tmp.pdb -o tmp_dssp.txt
# real    0m0,239s
# user    0m0,559s
# sys     0m0,040s

with open("./tmp_dssp.txt") as f:
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

start_time = time.time()
print('Computing DSSP')
SS_list = DSSP.compute_DSSP(test)
print(f'DSSP computed in {time.time() - start_time:.3f} ms')


def print_align_seq(seq_1, seq_2, line_len=80):
    """Print the aligned sequences with a line length of 80 characters.

    Parameters
    ----------
    seq_1 : str
        First sequence
    seq_2 : str
        Second sequence
    line_len : int, optional
        Length of the line, by default 80

    Returns
    -------
    None

    """

    sim_seq = ""
    for i in range(len(seq_1)):

        if seq_1[i] == seq_2[i]:
            sim_seq += "*"
            continue
        sim_seq += " "

    for i in range(1 + len(seq_1) // line_len):
        print(seq_1[i * line_len : (i + 1) * line_len])
        print(sim_seq[i * line_len : (i + 1) * line_len])
        print(seq_2[i * line_len : (i + 1) * line_len])
        print("\n")

    identity = 0
    similarity = 0
    for char in sim_seq:
        if char == "*":
            identity += 1
        if char in ["|", "*"]:
            similarity += 1

    len_1 = len(seq_1.replace("-", ""))
    len_2 = len(seq_2.replace("-", ""))

    print(f"Identity seq1: {identity / len_1 * 100:.2f}%")
    print(f"Identity seq2: {identity / len_2 * 100:.2f}%")

    print(f"Similarity seq1: {similarity / len_1 * 100:.2f}%")
    print(f"Similarity seq2: {similarity / len_2 * 100:.2f}%")

    return

print_align_seq(SS_list[0]['A'], dssp_seq, line_len=80)