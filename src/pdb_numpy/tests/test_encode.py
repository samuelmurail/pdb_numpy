#!/usr/bin/env python3
# coding: utf-8

"""
Tests for _alignement functions
"""

from pdb_numpy.format import encode_cython as encode


def test_encode_pure():
    for value in range(1000):
        s = encode.encode_pure(digits=encode.digits_upper, value=value)
        d = encode.decode_pure_Cap(s=s.encode('utf-8'))
        assert d == value

    def recycle4(value, encoded):
        s = encode.hy36encode(width=4, value=value)
        assert s == encoded
        d = encode.hy36decode(width=4, s=s)
        assert d == value

    assert encode.hy36decode(width=4, s="    ") == 0
    assert encode.hy36decode(width=4, s="  -0") == 0
    recycle4(-999, "-999")
    recycle4(-78, " -78")
    recycle4(-6, "  -6")
    recycle4(0, "   0")
    recycle4(9999, "9999")
    recycle4(10000, "A000")
    recycle4(10001, "A001")
    recycle4(10002, "A002")
    recycle4(10003, "A003")
    recycle4(10004, "A004")
    recycle4(10005, "A005")
    recycle4(10006, "A006")
    recycle4(10007, "A007")
    recycle4(10008, "A008")
    recycle4(10009, "A009")
    recycle4(10010, "A00A")
    recycle4(10011, "A00B")
    recycle4(10012, "A00C")
    recycle4(10013, "A00D")
    recycle4(10014, "A00E")
    recycle4(10015, "A00F")
    recycle4(10016, "A00G")
    recycle4(10017, "A00H")
    recycle4(10018, "A00I")
    recycle4(10019, "A00J")
    recycle4(10020, "A00K")
    recycle4(10021, "A00L")
    recycle4(10022, "A00M")
    recycle4(10023, "A00N")
    recycle4(10024, "A00O")
    recycle4(10025, "A00P")
    recycle4(10026, "A00Q")
    recycle4(10027, "A00R")
    recycle4(10028, "A00S")
    recycle4(10029, "A00T")
    recycle4(10030, "A00U")
    recycle4(10031, "A00V")
    recycle4(10032, "A00W")
    recycle4(10033, "A00X")
    recycle4(10034, "A00Y")
    recycle4(10035, "A00Z")
    recycle4(10036, "A010")
    recycle4(10046, "A01A")
    recycle4(10071, "A01Z")
    recycle4(10072, "A020")
    recycle4(10000 + 36**2 - 1, "A0ZZ")
    recycle4(10000 + 36**2, "A100")
    recycle4(10000 + 36**3 - 1, "AZZZ")
    recycle4(10000 + 36**3, "B000")
    recycle4(10000 + 26 * 36**3 - 1, "ZZZZ")
    recycle4(10000 + 26 * 36**3, "a000")
    recycle4(10000 + 26 * 36**3 + 35, "a00z")
    recycle4(10000 + 26 * 36**3 + 36, "a010")
    recycle4(10000 + 26 * 36**3 + 36**2 - 1, "a0zz")
    recycle4(10000 + 26 * 36**3 + 36**2, "a100")
    recycle4(10000 + 26 * 36**3 + 36**3 - 1, "azzz")
    recycle4(10000 + 26 * 36**3 + 36**3, "b000")
    recycle4(10000 + 2 * 26 * 36**3 - 1, "zzzz")

    #
    def recycle5(value, encoded):
        s = encode.hy36encode(width=5, value=value)
        assert s == encoded
        d = encode.hy36decode(width=5, s=s)
        assert d == value

    #
    assert encode.hy36decode(width=5, s="     ") == 0
    assert encode.hy36decode(width=5, s="   -0") == 0
    recycle5(-9999, "-9999")
    recycle5(-123, " -123")
    recycle5(-45, "  -45")
    recycle5(-6, "   -6")
    recycle5(0, "    0")
    recycle5(12, "   12")
    recycle5(345, "  345")
    recycle5(6789, " 6789")
    recycle5(99999, "99999")
    recycle5(100000, "A0000")
    recycle5(100010, "A000A")
    recycle5(100035, "A000Z")
    recycle5(100036, "A0010")
    recycle5(100046, "A001A")
    recycle5(100071, "A001Z")
    recycle5(100072, "A0020")
    recycle5(100000 + 36**2 - 1, "A00ZZ")
    recycle5(100000 + 36**2, "A0100")
    recycle5(100000 + 36**3 - 1, "A0ZZZ")
    recycle5(100000 + 36**3, "A1000")
    recycle5(100000 + 36**4 - 1, "AZZZZ")
    recycle5(100000 + 36**4, "B0000")
    recycle5(100000 + 2 * 36**4, "C0000")
    recycle5(100000 + 26 * 36**4 - 1, "ZZZZZ")
    recycle5(100000 + 26 * 36**4, "a0000")
    recycle5(100000 + 26 * 36**4 + 36 - 1, "a000z")
    recycle5(100000 + 26 * 36**4 + 36, "a0010")
    recycle5(100000 + 26 * 36**4 + 36**2 - 1, "a00zz")
    recycle5(100000 + 26 * 36**4 + 36**2, "a0100")
    recycle5(100000 + 26 * 36**4 + 36**3 - 1, "a0zzz")
    recycle5(100000 + 26 * 36**4 + 36**3, "a1000")
    recycle5(100000 + 26 * 36**4 + 36**4 - 1, "azzzz")
    recycle5(100000 + 26 * 36**4 + 36**4, "b0000")
    recycle5(100000 + 2 * 26 * 36**4 - 1, "zzzzz")
    #
    for width in [4, 5]:
        for value in [-(10 ** (width - 1)), 10**width + 2 * 26 * 36 ** (width - 1)]:
            try:
                encode.hy36encode(width=width, value=value)
            except (ValueError, RuntimeError) as e:
                assert str(e) == "value out of range."
            else:
                raise RuntimeError("Exception expected.")

    # Remove "40A0" and "410B0" from expected ValueError
    # Some correction introduced to support openmm pdb file format
    # made the previous values valid.

    for width, ss in [
        (4, ["", "    0", " abc", "abc-", "A=BC", "40a0"]),
        (5, ["", "     0", " abcd", "ABCD-", "a=bcd", "410b0"]),
    ]:
        for s in ss:
            print(s, ':')
            try:
                tmp = encode.hy36decode(width, s=s)
                print('tmp', tmp)
            except (ValueError, RuntimeError) as e:
                assert str(e) == "invalid number literal."
            else:
                raise RuntimeError(f"Exception expected for {s}.")
