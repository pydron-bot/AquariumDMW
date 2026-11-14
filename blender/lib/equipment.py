"""Hardware: the tank's black trim, the RGB+white LED bar on its bracket,
a green sponge filter, two small black internal filters and their cables.
Tank space (see util.py). The glass itself is drawn by three.js.
"""
import math

import numpy as np

from . import textures as T
from .util import Part, catmull, tube

TAU = 2 * math.pi

# Interior half-extents; glass 4 mm.
HX, HY, H = 0.25, 0.1275, 0.31
G = 0.004


def box(p, mat, lo, hi, color=(1, 1, 1)):
    """Axis-aligned box from corners, 6 faces with per-face UVs."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for q in quads:
        p.add([v[i] for i in q], [[0, 1, 2, 3]], [(0, 0), (1, 0), (1, 1), (0, 1)], mat, color)


def cyl(p, mat, c, r, z0, z1, sides=24, r1=None, color=(1, 1, 1)):
    pts = [(c[0], c[1], z0), (c[0], c[1], z1)]
    v, f, uv = tube(pts, [r, r if r1 is None else r1], sides, cap=True)
    p.add(v, f, uv, mat, color)
    # bottom cap
    n = len(v)
    centre = np.array([[c[0], c[1], z0]])
    ring = list(range(sides + 1))
    p.add(np.r_[v[: sides + 1], centre], [[sides + 1, ring[k + 1], ring[k]] for k in range(sides)],
          uv[: sides + 1] + [(0.5, 0.5)], mat, color)


def tank_trim():
    """Black plastic: bottom frame and top rim with a centre brace."""
    p = Part()
    ox, oy = HX + G, HY + G
    t = 0.008
    # bottom frame (wraps the base, 2.4 cm tall at the front)
    box(p, "trim", (-ox - t, -oy - t, -0.012), (ox + t, -oy, 0.02))
    box(p, "trim", (-ox - t, oy, -0.012), (ox + t, oy + t, 0.02))
    box(p, "trim", (-ox - t, -oy, -0.012), (-ox, oy, 0.02))
    box(p, "trim", (ox, -oy, -0.012), (ox + t, oy, 0.02))
    box(p, "trim", (-ox, -oy, -0.012), (ox, oy, -0.004))
    # top rim (2 cm) with an inward lip that holds the lid
    z0, z1 = H - 0.016, H + 0.006
    box(p, "trim", (-ox - t, -oy - t, z0), (ox + t, -oy, z1))
    box(p, "trim", (-ox - t, oy, z0), (ox + t, oy + t, z1))
    box(p, "trim", (-ox - t, -oy, z0), (-ox, oy, z1))
    box(p, "trim", (ox, -oy, z0), (ox + t, oy, z1))
    lip = 0.01
    box(p, "trim", (-ox, -oy, H - 0.004), (ox, -oy + lip, H))
    box(p, "trim", (-ox, oy - 0.03, H - 0.004), (ox, oy, H))  # rear strip (cut-outs for cables)
    box(p, "trim", (-ox, -oy, H - 0.004), (-ox + lip, oy, H))
    box(p, "trim", (ox - lip, -oy, H - 0.004), (ox, oy, H))
    return p


LED_ROWS = 3
LED_COLS = 72


def light_bar():
    """The LED bar ~9 cm above the rim on a centre-back bracket. The LED
    face is its own material with an emissive texture (see led_texture)."""
    p = Part()
    z = H + 0.085
    Lb, D, Tk = 0.465, 0.07, 0.011
    box(p, "light_body", (-Lb / 2, -D / 2, z), (Lb / 2, D / 2, z + Tk))
    # end caps
    for sx in (-1, 1):
        box(p, "light_body", (sx * Lb / 2 - 0.004 * (sx > 0), -D / 2 - 0.002, z - 0.002),
            (sx * Lb / 2 + 0.004 * (sx < 0), D / 2 + 0.002, z + Tk + 0.002))
    # LED face (just under the body)
    face = [(-Lb / 2 + 0.006, -D / 2 + 0.006, z - 0.0005), (Lb / 2 - 0.006, -D / 2 + 0.006, z - 0.0005),
            (Lb / 2 - 0.006, D / 2 - 0.006, z - 0.0005), (-Lb / 2 + 0.006, D / 2 - 0.006, z - 0.0005)]
    p.add(face, [[0, 3, 2, 1]], [(0, 0), (1, 0), (1, 1), (0, 1)], "led")
    # bracket: clamp on the back rim, a post, an arm to the bar
    by = HY + G + 0.004
    box(p, "light_body", (-0.018, by - 0.006, H - 0.02), (0.018, by + 0.012, H + 0.012))
    box(p, "metal", (-0.004, by - 0.002, H + 0.012), (0.004, by + 0.006, z))
    box(p, "light_body", (-0.022, -0.01, z + Tk), (0.022, by + 0.006, z + Tk + 0.008))
    return p


def led_texture(out, W=1024, H=64):
    """Emissive map for the LED face: three rows of 5050 LEDs, mostly cool
    white with red, green and blue mixed in like the real bar."""
    r = np.random.default_rng(9)
    img = np.zeros((H, W, 3))
    img[:] = (14, 14, 16)
    seq = ["w", "w", "r", "w", "b", "w", "g", "w", "w", "b", "w", "r"]
    colors = {"w": (255, 250, 240), "r": (255, 40, 40), "g": (40, 255, 80), "b": (60, 80, 255)}
    for row in range(LED_ROWS):
        cy = (row + 0.5) / LED_ROWS * H
        for k in range(LED_COLS):
            cx = (k + 0.5 + (0.5 if row == 1 else 0)) / LED_COLS * W
            c = colors[seq[(k + row * 5) % len(seq)]]
            yy, xx = np.mgrid[0:H, max(0, int(cx) - 8):min(W, int(cx) + 9)]
            d = np.sqrt((xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2)
            core = np.clip(1.3 - d / 3.2, 0, 1)[..., None]
            sl = img[:, max(0, int(cx) - 8):min(W, int(cx) + 9)]
            img[:, max(0, int(cx) - 8):min(W, int(cx) + 9)] = np.maximum(sl, core * np.array(c))
    T.save(out / "led.png", np.clip(img, 0, 255).astype(np.uint8))


def sponge_filter(x=-0.168, y=0.05):
    """Green base, dark sponge, green lift tube with clips, outlet elbow."""
    p = Part()
    base_z = 0.03
    cyl(p, "green_plastic", (x, y), 0.03, base_z - 0.004, base_z + 0.004)
    cyl(p, "sponge", (x, y), 0.023, base_z + 0.004, base_z + 0.068, sides=28)
    cyl(p, "green_plastic", (x, y), 0.009, base_z + 0.064, base_z + 0.076)
    # lift tube up to the surface, elbow, short spout
    pts = [(x, y, base_z + 0.07), (x, y, 0.24), (x, y, 0.262), (x + 0.006, y - 0.004, 0.272),
           (x + 0.02, y - 0.012, 0.274)]
    v, f, uv = tube(np.array(pts), [0.0068] * 5, 18, cap=False)
    p.add(v, f, uv, "green_tube")
    # suction-cup clips on the tube
    for zc in (0.12, 0.2):
        cyl(p, "green_plastic", (x, y), 0.0085, zc, zc + 0.008)
    # airline into the top
    air = [(x + 0.0, y + 0.004, 0.262), (x - 0.003, y + 0.02, 0.29), (x - 0.01, y + 0.06, 0.31),
           (x - 0.02, y + 0.11, 0.33), (x - 0.03, y + 0.15, 0.29), (x - 0.04, y + 0.17, 0.1)]
    air = catmull(air, 8)
    v, f, uv = tube(air, [0.0022] * len(air), 6, cap=False)
    p.add(v, f, uv, "cable")
    return p


def internal_filters():
    """Two small black submersible filters on the back glass, upper left."""
    p = Part()
    for k, x in enumerate((-0.222, -0.19)):
        y = HY - 0.02
        z0 = 0.165
        box(p, "black_plastic", (x - 0.014, y - 0.015, z0), (x + 0.014, y + 0.015, z0 + 0.045))
        # intake grille band and outlet nozzle
        box(p, "black_plastic", (x - 0.0145, y - 0.0155, z0 + 0.004), (x + 0.0145, y + 0.0155, z0 + 0.012))
        nz = [(x + 0.014, y - 0.004, z0 + 0.03), (x + 0.024, y - 0.004, z0 + 0.03)]
        v, f, uv = tube(np.array(nz), [0.003, 0.0026], 10)
        p.add(v, f, uv, "black_plastic")
        # suction cups on the back glass
        for zc in (z0 + 0.012, z0 + 0.034):
            pts = [(x, HY - 0.005, zc), (x, HY - 0.0005, zc)]
            v, f, uv = tube(np.array(pts), [0.004, 0.0065], 14)
            p.add(v, f, uv, "suction_cup")
        cable = catmull([(x, y, z0 + 0.045), (x + 0.004, y + 0.004, 0.24), (x + 0.008, y + 0.01, 0.3),
                         (x + 0.012, HY + 0.02, 0.335), (x + 0.02, HY + 0.05, 0.25), (x + 0.03, HY + 0.06, 0.0)], 8)
        v, f, uv = tube(cable, [0.0021] * len(cable), 6, cap=False)
        p.add(v, f, uv, "cable")
    return p
