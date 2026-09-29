import numpy as np
import pytest

from trustsr.trust import dihedral, undo_dihedral, ndvi_moments


@pytest.mark.parametrize('index', range(8))
def test_dihedral_inverse(index):
    image = np.arange(3 * 5 * 7).reshape(3, 5, 7)
    np.testing.assert_array_equal(undo_dihedral(dihedral(image, index), index), image)


def test_ndvi_moments_are_computed_per_run():
    a = np.zeros((4, 1, 1), dtype=np.float32)
    b = a.copy()
    a[0], a[3] = .2, .6
    b[0], b[3] = .6, .2
    mean, std = ndvi_moments([a, b], red_index=0, nir_index=3)
    np.testing.assert_allclose(mean, [[0]], atol=1e-6)
    np.testing.assert_allclose(std, [[.5]], atol=1e-6)
