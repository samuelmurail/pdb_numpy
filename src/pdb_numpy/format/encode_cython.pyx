"""
Encodes and decodes numbers as strings.
Taken From :
https://raw.githubusercontent.com/cctbx/cctbx_project/master/iotbx/pdb/hybrid_36.py

               Prototype/reference implementation for
                      encoding and decoding
                     atom serial numbers and
                     residue sequence numbers
                          in PDB files.

PDB ATOM and HETATM records reserve columns 7-11 for the atom serial
number. This 5-column number is used as a reference in the CONECT
records, which also reserve exactly five columns for each serial
number.

With the decimal counting system only up to 99999 atoms can be stored
and uniquely referenced in a PDB file. A simple extension to enable
processing of more atoms is to adopt a counting system with more than
ten digits. To maximize backward compatibility, the counting system is
only applied for numbers greater than 99999. The "hybrid-36" counting
system implemented in this file is:

  ATOM      1
  ...
  ATOM  99999
  ATOM  A0000
  ATOM  A0001
  ...
  ATOM  A0009
  ATOM  A000A
  ...
  ATOM  A000Z
  ATOM  ZZZZZ
  ATOM  a0000
  ...
  ATOM  zzzzz

I.e. the first 99999 serial numbers are represented as usual. The
following atoms use a base-36 system (10 digits + 26 letters) with
upper-case letters. 43670016 (26*36**4) additional atoms can be
numbered this way. If there are more than 43770015 (99999+43670016)
atoms, a base-36 system with lower-case letters is used, allowing for
43670016 additional atoms. I.e. in total 87440031 (99999+2*43670016)
atoms can be stored and uniquely referenced via CONECT records.

The counting system is designed to avoid lower-case letters until the
range of numbers addressable by upper-case letters is exhausted.
Importantly, with this counting system the distinction between
"traditional" and "extended" PDB files becomes evident only if there
are more than 99999 atoms to be stored. Programs that are
updated to support the hybrid-36 counting system will continue to
interoperate with programs that do not as long as there are less than
100000 atoms.

PDB ATOM and HETATM records also reserve columns 23-26 for the residue
sequence number. This 4-column number is used as a reference in other
record types (SSBOND, LINK, HYDBND, SLTBRG, CISPEP), which also reserve
exactly four columns for each sequence number.

With the decimal counting system only up to 9999 residues per chain can
be stored and uniquely referenced in a PDB file. If the hybrid-36
system is adopted, 1213056 (26*36**3) additional residues can be
numbered using upper-case letters, and the same number again using
lower-case letters. I.e. in total each chain may contain up to 2436111
(9999+2*1213056) residues that can be uniquely referenced from the
other record types given above.

The implementation in this file should run with Python 2.6 or higher.
There are no other requirements. Run this script without arguments to
obtain usage examples.

Note that there are only about 60 lines of "real" code. The rest is
documentation and unit tests.

To update an existing program to support the hybrid-36 counting system,
simply replace the existing read/write source code for integer values
with equivalents of the hy36decode() and hy36encode() functions below.

This file is unrestricted Open Source (cctbx.sf.net).
Please send corrections and enhancements to cctbx@cci.lbl.gov .

See also:
  http://cci.lbl.gov/hybrid_36/
  http://www.pdb.org/ "Dictionary & File Formats"

Ralf W. Grosse-Kunstleve, Feb 2007.

The code has been rewritten in cython by Samuel Murail to accelerate it.
"""

#cython: embedsignatures=True 

import cython
from libc.string cimport strcpy, strlen

cdef int convert_low(char c):
    if c >= 48 and c <=57: #'0':48 '9':57
        return c - 48
    elif c >= 97 and c <=122: #'a':97 'z':122
        #print(chr(c), c - 87)
        return c - 87 # -97 'a' + 10
    raise ValueError("invalid number literal.")

cdef int convert_Cap(char c):
    if c >= 48 and c <=57: #'0':48 '9':57
        return c - 48
    elif c >= 65 and c <=90: #'A':65 'Z':90
        #print(chr(c), c - 87)
        return c - 55 # -65 'A' + 10
    raise ValueError("invalid number literal.")
    

cdef int decode_pure_Low(char * s):
    """ Decodes the string s using value associations for each character
    """
    cdef:
        int result = 0, n = 36
        char c
    
    for c in s:
        result *= n
        result += convert_low(c)
    return result


cpdef int decode_pure_Cap(char * s):
    "decodes the string s using value associations for each character"
    cdef:
        int result = 0, n = 36
        char c
    
    for c in s:
        result *= n
        result += convert_Cap(c)
    return result

cdef bint isDigit(char c):
    if c >= 48 and c <=57: #'0':48 '9':57
        return True
    else:
        return False

cdef bint isCapLetter(char c):
    if c >= 65 and c <=90: #'A':65 'Z':90
        return True
    else:
        return False

cdef bint isLowLetter(char c):
    if c >= 97 and c <=122: #'a':97 'z':122   
        return True
    else:
        return False
        
def hy36decode(int width, str s):
    """ Decodes base-10/upper-case base-36/lower-case base-36 hybrid.
    """

    tmp = s.encode('utf-8')

    cdef:
        char f
        char * number_str
    number_str = tmp

    if strlen(number_str) == width:
        f = number_str[0]
        if f == b"-" or f == b" " or isDigit(f):
            try:
                return int(s)
            except ValueError:
                pass
            if s == " " * width:
                return 0
        elif isCapLetter(f):
            try:
                return (
                    decode_pure_Cap(number_str)
                    - 10 * 36 ** (width - 1)
                    + 10**width
                )
            except KeyError:
                pass
        elif isLowLetter(f):
            try:
                return (
                    decode_pure_Low(s=number_str)
                    + 16 * 36 ** (width - 1)
                    + 10**width
                )
            except KeyError:
                pass
    raise ValueError("invalid number literal.")

digits_upper = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits_lower = "0123456789abcdefghijklmnopqrstuvwxyz"

def encode_pure(str digits, int value):
    """ Encodes value using the given digits.
    """
    cdef:
        int n
        list result = []
        int rest

    assert value >= 0

    if value == 0:
        return digits[0]
    
    n = 36
    result = []

    while value != 0:
        rest = value // n
        result.append(digits[value - rest * n])
        value = rest
    
    result.reverse()
    return "".join(result)


def hy36encode(int width, int value):
    """ Encodes value as base-10/upper-case base-36/lower-case
    base-36 hybrid.
    """
    cdef:
        double i = value
        
    
    if i >= 1 - 10 ** (width - 1):

        if i < 10**width:
            return ("%%%dd" % width) % i
        
        i -= 10**width

        if i < 26 * 36 ** (width - 1):
            i += 10 * 36 ** (width - 1)
            return encode_pure(digits_upper, int(i))
        
        i -= 26 * 36 ** (width - 1)
        if i < 26 * 36 ** (width - 1):
            i += 10 * 36 ** (width - 1)
            return encode_pure(digits_lower, int(i))
        
    raise ValueError("value out of range.")
