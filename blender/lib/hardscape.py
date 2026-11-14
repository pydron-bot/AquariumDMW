"""Driftwood, rocks and pebbles, built in tank space (origin = centre of the
tank floor, Blender +Y = toward the back glass, Z up), laid out after the
photos: a big gnarled stump left of centre with a tall arm, branches
climbing to the back right and reaching the surface, a low log front right.
"""
import math

import numpy as np

from .plants import pennywort_vine
from .util import Part, catmull, fbm3, tube


def substrate_height(x, y):
    """Sand level (m) at tank-space x, y (Blender y: + is back). The same
    formula is in src/tank/substrate.ts; keep them in step."""
    back = (y + 0.1275) / 0.255
    return 0.028 + 0.024 * back + 0.008 * math.exp(-((x + 0.06) ** 2) / 0.004)


WOOD = {
    # name: [(x, y, z, radius), ...]
    "stump": [(-0.07, 0.012, 0.03, 0.036), (-0.062, 0.004, 0.08, 0.031), (-0.056, 0.012, 0.128, 0.022),
              (-0.064, 0.022, 0.168, 0.015), (-0.076, 0.03, 0.198, 0.009)],
    "root_l": [(-0.125, -0.025, 0.028, 0.012), (-0.1, -0.012, 0.04, 0.02), (-0.07, 0.0, 0.055, 0.028)],
    "root_r": [(-0.012, -0.03, 0.03, 0.013), (-0.035, -0.012, 0.045, 0.022), (-0.06, 0.0, 0.06, 0.028)],
    "root_b": [(-0.1, 0.07, 0.04, 0.012), (-0.085, 0.04, 0.05, 0.02), (-0.065, 0.015, 0.065, 0.026)],
    "upright": [(-0.03, 0.025, 0.05, 0.021), (-0.004, 0.04, 0.115, 0.016), (0.012, 0.052, 0.178, 0.012),
                (0.018, 0.06, 0.218, 0.007)],
    "climber": [(-0.035, 0.0, 0.065, 0.022), (0.025, 0.028, 0.108, 0.018), (0.085, 0.05, 0.158, 0.015),
                (0.128, 0.068, 0.205, 0.011), (0.152, 0.08, 0.232, 0.007)],
    "reach": [(0.04, 0.035, 0.115, 0.012), (0.105, 0.075, 0.198, 0.01), (0.158, 0.098, 0.262, 0.008),
              (0.176, 0.104, 0.284, 0.006)],
    "log": [(-0.015, -0.022, 0.048, 0.019), (0.045, 0.0, 0.066, 0.016), (0.098, 0.028, 0.082, 0.013),
            (0.135, 0.05, 0.088, 0.008)],
    "twig": [(-0.058, 0.012, 0.15, 0.01), (-0.095, 0.002, 0.178, 0.007), (-0.122, -0.008, 0.19, 0.004)],
}


def driftwood():
    """Spider-wood-like branches: tubes with twisting ridges and knots."""
    p = Part()
    for idx, (name, pts) in enumerate(WOOD.items()):
        arr = np.array(pts, dtype=float)
        path = catmull(arr[:, :3], 9)
        radii = np.interp(np.linspace(0, len(arr) - 1, len(path)), np.arange(len(arr)), arr[:, 3])
        seed = 300 + idx

        def bump(pos, d, i, k, seed=seed, n=len(path)):
            q = (pos + d * 0.01)[None, :] * 70
            knot = fbm3(q, 3, seed)[0]
            a = 2 * math.pi * k / 14
            ridge = math.sin(3 * a + i * 0.35 + seed) * 0.12 + math.sin(7 * a - i * 0.6) * 0.05
            return max(0.45, 1 + 0.32 * knot + ridge)

        v, f, uv = tube(path, radii, 14, uv_scale=1.0, bump=bump)
        tone = 0.85 + 0.3 * ((idx * 37) % 10) / 10
        p.add(v, f, uv, "bark", (tone, tone, tone))
    return p


def lump(center, radii, seed, rough=0.25, freq=40.0, nu=22, nv=14, sink=0.3):
    """A displaced ellipsoid (rock or pebble), sunk into the sand."""
    c = np.asarray(center, dtype=float)
    R = np.asarray(radii, dtype=float)
    verts, uvs = [], []
    for j in range(nv + 1):
        th = math.pi * j / nv
        for i in range(nu + 1):
            ph = 2 * math.pi * i / nu
            d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            n = fbm3((d * freq * R.mean() + seed)[None, :], 4, seed)[0]
            verts.append(c + d * R * (1 + rough * n))
            uvs.append((i / nu * 2, j / nv))
    v = np.array(verts)
    v[:, 2] -= R[2] * sink
    faces = []
    w = nu + 1
    for j in range(nv):
        for i in range(nu):
            a = j * w + i
            faces.append([a, a + w, a + w + 1, a + 1])
    return v, faces, uvs


def rocks():
    p = Part()
    dark = [((-0.03, -0.035, 0.035), (0.03, 0.024, 0.022)), ((0.005, 0.0, 0.045), (0.026, 0.022, 0.026)),
            ((-0.1, 0.03, 0.05), (0.024, 0.02, 0.02)), ((0.06, 0.02, 0.045), (0.02, 0.018, 0.016))]
    for k, (c, r) in enumerate(dark):
        z = substrate_height(c[0], c[1])
        v, f, uv = lump((c[0], c[1], z), r, 400 + k, rough=0.35, freq=60)
        p.add(v, f, uv, "rock_dark", (1, 1, 1))
    pebbles = [((-0.17, -0.075), (0.018, 0.014, 0.008), (0.9, 0.92, 0.95)),
               ((-0.06, -0.085), (0.016, 0.013, 0.008), (1.15, 1.12, 1.05)),
               ((0.005, -0.06), (0.013, 0.012, 0.007), (0.75, 0.82, 0.8)),
               ((0.12, -0.075), (0.019, 0.015, 0.009), (0.85, 0.9, 0.88)),
               ((-0.205, -0.04), (0.009, 0.008, 0.005), (1.05, 1.0, 0.95)),
               ((0.045, -0.095), (0.008, 0.006, 0.004), (1.1, 1.05, 1.0))]
    for k, ((x, y), r, tint) in enumerate(pebbles):
        z = substrate_height(x, y)
        v, f, uv = lump((x, y, z), r, 450 + k, rough=0.05, freq=10, sink=0.35)
        p.add(v, f, uv, "pebble", tint)
    return p


PENNYWORT_PATHS = [
    [(0.018, 0.06, 0.212), (0.008, 0.04, 0.245), (-0.04, 0.01, 0.266), (-0.1, -0.01, 0.268), (-0.16, -0.03, 0.266)],
    [(0.15, 0.08, 0.228), (0.14, 0.04, 0.258), (0.08, -0.01, 0.268), (0.02, -0.05, 0.268), (-0.03, -0.07, 0.265)],
    [(-0.076, 0.03, 0.196), (-0.09, 0.01, 0.235), (-0.13, -0.03, 0.264), (-0.2, -0.05, 0.268)],
    [(0.2, 0.02, 0.045), (0.2, 0.01, 0.15), (0.205, -0.02, 0.24), (0.17, -0.06, 0.266), (0.09, -0.09, 0.268)],
    [(0.06, 0.07, 0.07), (0.07, 0.08, 0.17), (0.04, 0.09, 0.25), (-0.05, 0.08, 0.268), (-0.12, 0.09, 0.266)],
]


def pennywort():
    p = Part()
    for k, path in enumerate(PENNYWORT_PATHS):
        p.merge(pennywort_vine(600 + k, path))
    return p
