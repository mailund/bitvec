"""Testing bit vectors."""

from bitvec import BitVec


def test_construction() -> None:
    """Test that we can construct a bit vector."""
    bv = BitVec(33)
    assert len(bv.bits) == 2, "It takes two words to represent 33 bits"


def test_setters_and_getters() -> None:
    """Test that we can change bits."""
    bv = BitVec(4)
    assert len(bv.bits) == 1, "It takes one word to represent 4 bits"

    for i in range(4):
        assert not bv[i], "The bits should be all zeros now"

    for i in range(4):
        bv[i] = (i % 2 == 0)  # Even bits should be true...
    for i in range(4):
        assert bv[i] == (i % 2 == 0), "The even indices should be set"

    for i in range(4):
        bv[i] = (i % 2 != 0)  # Odd bits should be true...
    for i in range(4):
        assert bv[i] != (i % 2 == 0), "The odd indices should be set"


def test_rank() -> None:
    """Test that we can get the correct rank (sort of)."""
    bv = BitVec(100)
    # Set odd index
    for i in range(100):
        bv[i] = bool(i % 2 != 0)

    # the rank should always be i//2 when only odd indices are set
    #       01 01 01
    # rank: 00 11 22
    for i in range(100):
        print(i, bv[i])
        assert bv.rank1(i) == i//2


if __name__ == '__main__':
    import pytest
    pytest.main()
