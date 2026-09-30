import numpy as np
from math import erf, sqrt, log, pi

# (1) NDVI drop of a mixed 10 m pixel vs true bare-soil fraction f. ILLUSTRATIVE endmembers (assumed, not measured).
ev = (0.03, 0.40)   # (red, nir) dense forest, assumed
eb = (0.15, 0.20)   # (red, nir) bare soil/rock, assumed
nd = lambda r, n: (n - r) / (n + r)
print("NDVI veg %.3f bare %.3f" % (nd(*ev), nd(*eb)))
for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    r = (1 - f) * ev[0] + f * eb[0]
    n = (1 - f) * ev[1] + f * eb[1]
    print("f=%.1f  NDVI_10m=%.3f drop=%.3f  linear-in-f drop would be %.3f  (N-R)=%.3f" % (f, nd(r, n), nd(*ev) - nd(r, n), f * (nd(*ev) - nd(*eb)), n - r))


# (2) box vs Gaussian-PSF block fraction, 1-D straight edge, 10 m blocks. sigma from MTF(Nyquist)
def sigma_from_mtf(m):
    fN = 1 / (2 * 10.0)
    return sqrt(-log(m) / (2 * pi ** 2 * fN ** 2))


Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
x0 = np.linspace(-15, 25, 4001)
mixed = (x0 > 0) & (x0 < 10)
for m in (0.15, 0.2, 0.3):
    s = sigma_from_mtf(m)
    errs = []
    for x in x0:
        fb = min(max((10 - x) / 10, 0), 1)
        fp = 1 - Phi((x - 5) / s)
        errs.append(abs(fb - fp))
    errs = np.array(errs)
    print("MTF@Nyq=%.2f sigma=%.2f m  mean|f_box-f_psf| over offsets in [-15,25]=%.3f max=%.3f mean over blocks straddling the edge=%.3f" % (m, s, errs.mean(), errs.max(), errs[mixed].mean()))
print("box sigma equivalent = %.2f m, box MTF at Nyquist = %.3f" % (10 / sqrt(12), np.sin(np.pi / 2) / (np.pi / 2)))

# (3) allocation: symmetric difference for a block with n true sub-pixels among 16
for n in (2, 4, 8, 12, 14):
    f = n / 16
    blocky = n if n < 8 else 16 - n if n > 8 else 8
    rand = 2 * (n - n * n / 16)
    print("n=%d f=%.3f blocky(0.5 threshold) symdiff=%d   random-allocation E[symdiff]=%.2f  oracle=0" % (n, f, blocky, rand))
