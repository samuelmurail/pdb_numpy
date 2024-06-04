#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: linetrace=True
#cython: embedsignatures=True 

import cython
from libc.string cimport strcpy, strlen

cdef bint is_space (char c):
    if c == b" ":
        return True
    else:
        return False

cdef bint is_quote (char c):
    if c == b"'" or c ==b'"':
        return True
    else:
        return False

@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing.
def string_split(str line_raw):

    tmp = line_raw.encode('utf-8')

    cdef:
        int i=0, j=0, k=0, count=0, str_len
        char first_quote, * line
        list split_list = []
    
    line = tmp
    str_len = strlen(line)

    while (i<str_len) :
        
        while (i<str_len and is_space(line[i])):
            i += 1
                
        if is_quote(line[i]):
            first_quote = line[i]
            j = i
            i += 1
            while (i<str_len and line[i] != first_quote):
                i += 1
            i += 1
        else:
            j = i
            while (i<str_len and not is_space(line[i])):
                i += 1
        if i!= j:
            split_list.append(line[j:i].decode('UTF-8'))
        i+=1
    return split_list