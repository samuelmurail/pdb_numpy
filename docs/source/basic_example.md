
# Basic Usage

To use`pdb_numpy` in a project:

```python
import pdb_numpy
```

Create a `Coor()` object

## Loading a structure

You can either get the coordinates from the Protein Data Bank:

```python
coor_1hsg = pdb_numpy.Coor(pdb_id='1hsg')
```

Or load a local stored `.pdb` file:

```python
coor_1hsg = pdb_numpy.Coor('./1hsg.pdb')
```

or `.cif` file:

```python
coor_1hsg = pdb_numpy.Coor('./1hsg.cif')
```

## Selection of coordinates subset

You can extract a selection of coordinates, here we will use the `1hsg.pdb` PDB file and extract the coordinates of L-735,524 an inhibitor of the HIV proteases (resname MK1):

```python
# Select res_name MK1
lig_coor = coor_1hsg.select_atoms("resname MK1")
```

The obtain selection can be saved using the write_pdb() function:

```python
# Save the ligand coordinates
lig_coor.write_pdb('1hsg_lig.pdb')
```

For selection you can use the following keywords :
- `name` for atom name
- `altloc` for alternative location
- `resname` for residue name
- `chain` for chain ID
- `resid` residue number
- `residue` a unique residue number starting from 0
-  `x`, `y`, `z`, coordinates
- `occ` for occupation
- `beta` for beta factor

Selector can be combined, eg. to select residue names Proline and Alanine from chain A you can use:

```python
PRO_ALA_A_coor = coor_1hsg.select_atoms("resname PRO ALA and chain A")
```

The following combinator `and`, `or`, `not` and `within of` can be used.
Here we are selecting atoms of chain A and within 5.0 Å of chain B.

```python
interface_coor = coor_1hsg.select_atoms("chain A and within 5.0 of chain B")
```

Moreover different operators can be used as "==", "!=", ">", ">=", "<", "<=":

```python
Nter_coor = coor_1hsg.select_atoms("chain A and resid <= 10")
```


> **note:**
> To select protein atoms you can use the `protein` and `backbone` keywords.
>
> ```python
> prot_coor = coor_1hsg("protein and chain A")
> ```

