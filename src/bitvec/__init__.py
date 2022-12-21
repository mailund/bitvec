"""A small bit vector implementation."""

import numpy as np
import numpy.typing as npt

from typing import (
    Optional
)

_WORD_SIZE = 32


def _word_index(i: int) -> int:
    """Get the word that contains bit i."""
    return i // _WORD_SIZE


def _bit_index(i: int) -> int:
    """Get the word-offset for bit i."""
    return i % _WORD_SIZE


def _word_rank(w: np.int32, i: int) -> int:
    """Count number of set bits in the first i bits of w."""
    mask = (1 << i) - 1
    return (int(w) & mask).bit_count()


class BitVec:
    """A small bit vector implementation."""

    bits: npt.NDArray[np.int32]
    _ranks: Optional[npt.NDArray[np.int32]]

    def __init__(self, size: int) -> None:
        """Build a bit vector for `size` bits."""
        no_words = (size + _WORD_SIZE - 1) // _WORD_SIZE
        self.bits = np.zeros(no_words, dtype=np.int32)
        self._ranks = None

    def __getitem__(self, i: int) -> bool:
        """Get the bit at index i."""
        w = self.bits[_word_index(i)] >> _bit_index(i)
        return bool(0x1 & w)

    def __setitem__(self, i: int, b: bool) -> None:
        """Set the bit at index i to b."""
        self._ranks = None  # Modifying the bit vector invalidates the ranks
        w = self.bits[_word_index(i)]
        if b:
            w |= 1 << _bit_index(i)
        else:
            w &= ~(1 << _bit_index(i))
        self.bits[_word_index(i)] = w

    @property
    def ranks(self) -> npt.NDArray[np.int32]:
        """Get the 1-ranks after each word in bits."""
        if self._ranks is None:
            # Preprocessing for rank queries...
            self._ranks = np.zeros(len(self.bits) + 1, dtype=np.int32)
            for i in range(1, len(self._ranks)):
                self._ranks[i] = self._ranks[i-1] + \
                    int(self.bits[i-1]).bit_count()

        return self._ranks

    def rank0(self, i: int) -> int:
        """Get the zero-rank at index i."""
        return i - self.rank1(i)

    def rank1(self, i: int) -> int:
        """Get the one-rank at index i."""
        wi, bi = _word_index(i), _bit_index(i)
        return int(self.ranks[wi]) + _word_rank(self.bits[wi], bi)
