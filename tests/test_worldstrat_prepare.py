import io
import zipfile

import numpy as np
import pytest


def test_member_checks_crc_and_limits():
    from risk.worldstrat_prepare import decode_member
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('a.tif', b'genuine file bytes')
    data = buffer.getvalue()
    assert decode_member(data, 'a.tif', 100) == b'genuine file bytes'
    with pytest.raises(ValueError):
        decode_member(data, 'different.tif', 100)
    with pytest.raises(ValueError):
        decode_member(data, 'a.tif', 3)
    damaged = bytearray(data)
    damaged[14] ^= 1  # Local-header CRC must be checked, not trusted.
    with pytest.raises(ValueError, match='CRC'):
        decode_member(bytes(damaged), 'a.tif', 100)


def test_valid_land_crop_rejects_clouds_and_water():
    from risk.worldstrat_prepare import choose_crop
    scl = np.full((5, 5), 4)
    scl[0, :] = 8
    y, x = choose_crop(scl, np.ones_like(scl), 4, [4, 5, 7], .5)
    assert (y, x) == (1, 0)
    assert choose_crop(np.full((5, 5), 6), np.ones((5, 5)), 4, [4, 5, 7], .5) is None
    assert choose_crop(np.full((8, 8), 4), np.ones((8, 8)), 4, [4, 5, 7], .5) == (2, 2)


def test_harmonization_recovers_known_band_gains_without_inventing_detail():
    from risk.worldstrat_prepare import harmonize
    low = np.arange(16, dtype=float).reshape(1, 4, 4) / 100 + .1
    dn = (low - .02) / .001
    high = np.repeat(np.repeat(dn, 4, axis=1), 4, axis=2)
    fitted, coefficients = harmonize(low, high, 4, .99)
    np.testing.assert_allclose(fitted.reshape(1, 4, 4, 4, 4).mean((2, 4)), low, atol=1e-6)
    assert coefficients[0]['gain'] == pytest.approx(.001)
    assert coefficients[0]['offset'] == pytest.approx(.02)
    with pytest.raises(ValueError, match='variance'):
        harmonize(low, np.ones_like(high), 4, .99)
