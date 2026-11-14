"""Plant generators. Each builds a Part with the plant's base at the origin
(Blender Z up); three.js places them from its layout table and sways them
in the vertex shader by height above the base.

Leaves are real geometry (curved, cupped, rippled grids) with their own UV
square, so the painted leaf textures land the same way on every leaf.
Sizes are the real plants', in metres.
"""
import math

import numpy as np
from mathutils import Matrix, Vector

from .util import Part, catmull, frame, grid_faces, look_frame, tube

TAU = 2 * math.pi


def blade(length, width, shape=(0.45, 0.7), nu=6, nv=12, droop=0.4, cup=0.15, wave=0.0,
          wave_freq=5.0, side_bend=0.0, twist=0.0, seed=0, base_w=0.0, tip_w=0.0, fork=None):
    """A leaf along +Y, normal +Z. shape (a, b): width ~ v^a (1-v)^b.
    droop bends the blade down along its length (radians at the tip),
    cup raises the edges, wave ripples them (crypts, java fern)."""
    r = np.random.default_rng(seed)
    a, b = shape
    vmax = a / (a + b) if a + b > 0 else 0.5
    norm = (vmax ** a) * ((1 - vmax) ** b) if a + b > 0 else 1.0
    ph = r.uniform(0, TAU)
    verts, uvs = [], []
    # centreline, bent by droop (pitch) and side_bend (yaw)
    pos = np.zeros(3)
    centre, frames = [], []
    for j in range(nv + 1):
        v = j / nv
        th = droop * v ** 1.3
        yw = side_bend * v ** 1.5
        T = np.array([math.sin(yw) * math.cos(th), math.cos(yw) * math.cos(th), -math.sin(th)])
        if j > 0:
            pos = pos + T * (length / nv)
        X = np.array([math.cos(yw), -math.sin(yw), 0.0])
        N = np.cross(X, T)
        tw = twist * v
        X, N = X * math.cos(tw) + N * math.sin(tw), N * math.cos(tw) - X * math.sin(tw)
        centre.append(pos.copy())
        frames.append((X, N))
    for j in range(nv + 1):
        v = j / nv
        w = ((v ** a) * ((1 - v) ** b) / norm) if a + b > 0 else 1.0
        w = max(w, base_w * (1 - v)) if base_w else w
        w = w * (1 - tip_w) + tip_w * (1 - v) if tip_w else w
        hw = 0.5 * width * w
        X, N = frames[j]
        for i in range(nu + 1):
            u = i / nu
            s = (u - 0.5) * 2
            z = cup * width * s * s * 0.5
            z += wave * math.sin(wave_freq * TAU * v + ph + s * 0.8) * abs(s) ** 1.6 * min(1.0, v * 6)
            p = centre[j] + X * (s * hw) + N * z
            verts.append(p)
            uvs.append((u, v))
    faces = grid_faces(nu, nv)
    return np.array(verts), faces, uvs


def add_leaf(part, mat, m, color, **kw):
    v, f, uv = blade(**kw)
    p = np.c_[v, np.ones(len(v))] @ np.array(m).T
    part.add(p[:, :3], f, uv, mat, color)


def tint(r, base=1.0, var=0.12):
    k = base * r.uniform(1 - var, 1 + var)
    return (k, k * r.uniform(0.96, 1.04), k * r.uniform(0.94, 1.03))


def stem(part, pts, r0, r1, mat, sides=5, per=4, color=(1, 1, 1)):
    pts = catmull(pts, per) if len(pts) > 2 else np.asarray(pts, float)
    radii = np.linspace(r0, r1, len(pts))
    v, f, uv = tube(pts, radii, sides, uv_scale=0.2)
    part.add(v, f, uv, mat, color)
    return pts


def roots(part, r, origin, n, length, mat="roots_white"):
    for _ in range(n):
        d = np.array([r.normal(0, 0.6), r.normal(0, 0.6), -1.0])
        d /= np.linalg.norm(d)
        L = length * r.uniform(0.5, 1.2)
        # wandering, not straight: roots kink around grains and wood
        pts, p = [np.asarray(origin, float)], np.asarray(origin, float)
        for k in range(5):
            d = d + np.array([r.normal(0, 0.45), r.normal(0, 0.45), r.normal(-0.2, 0.2)])
            d /= np.linalg.norm(d)
            p = p + d * L / 5
            pts.append(p)
        stem(part, pts, 0.00045, 0.0002, mat, sides=4, per=4)


# ---------------------------------------------------------------- species

def anubias(seed, leaves=8, size=1.0):
    """Anubias barteri: creeping rhizome, glossy ovate leaves on petioles."""
    r = np.random.default_rng(seed)
    p = Part()
    rl = 0.05 * size
    rpts = [(-rl / 2, 0, 0.004), (-rl / 6, r.normal(0, 0.006), 0.006), (rl / 6, r.normal(0, 0.006), 0.005),
            (rl / 2, 0, 0.004)]
    rpts = stem(p, rpts, 0.0042 * size, 0.0032 * size, "rhizome", sides=7)
    for k in range(4):
        roots(p, r, rpts[int(len(rpts) * (k + 0.5) / 4)], 2, 0.025, "roots_white")
    for k in range(leaves):
        t = (k + 0.5) / leaves
        base = rpts[min(len(rpts) - 1, int(t * len(rpts)))] + np.array([0, 0, 0.002])
        yaw = r.uniform(0, TAU)
        elev = math.radians(r.uniform(35, 75))
        pl = r.uniform(0.015, 0.04) * size
        d = np.array([math.sin(yaw) * math.cos(elev), math.cos(yaw) * math.cos(elev), math.sin(elev)])
        tip = base + d * pl
        stem(p, [base, base + d * pl * 0.5 + np.array([0, 0, pl * 0.1]), tip], 0.0017, 0.0013, "stem_green",
             per=3, color=tint(r, 0.8))
        L = r.uniform(0.04, 0.062) * size * (0.75 if k % 3 == 0 else 1)
        m = frame(tip, yaw=-yaw, pitch=elev - math.radians(r.uniform(15, 35)), roll=r.uniform(-0.4, 0.4))
        add_leaf(p, "leaf_anubias", m, tint(r, r.uniform(0.75, 1.05)), length=L, width=L * r.uniform(0.6, 0.72),
                 shape=(0.5, 0.55), nu=6, nv=11, droop=r.uniform(0.1, 0.45), cup=0.12, wave=0.0006,
                 wave_freq=2.0, seed=seed * 50 + k, base_w=0.25)
    return p


def javafern(seed, leaves=9, size=1.0, trident=False):
    """Microsorum pteropus; trident: narrow three-lobed fronds."""
    r = np.random.default_rng(seed)
    p = Part()
    rpts = stem(p, [(-0.015, 0, 0.003), (0, r.normal(0, 0.004), 0.005), (0.015, 0, 0.003)], 0.003, 0.0025,
                "rhizome", sides=6)
    roots(p, r, rpts[len(rpts) // 2], 6, 0.03, "rhizome")
    for k in range(leaves):
        base = rpts[r.integers(len(rpts))]
        yaw = r.uniform(0, TAU)
        elev = math.radians(r.uniform(55, 82))
        L = r.uniform(0.08, 0.16) * size
        d = np.array([math.sin(yaw) * math.cos(elev), math.cos(yaw) * math.cos(elev), math.sin(elev)])
        pt = base + d * 0.012
        stem(p, [base, pt], 0.0015, 0.0012, "stem_green", color=tint(r, 0.7))
        col = tint(r, r.uniform(0.75, 1.05))
        if trident:
            for lobe, (dy, sc) in enumerate(((0, 1.0), (-0.45, 0.55), (0.45, 0.55))):
                if lobe and r.random() < 0.25:
                    continue
                m = frame(pt + d * (0 if lobe == 0 else L * 0.25), yaw=-yaw + dy, pitch=elev - 0.15 * (lobe > 0))
                add_leaf(p, "leaf_javafern", m, col, length=L * sc * 0.8, width=0.0075 * size,
                         shape=(0.25, 0.8), nu=3, nv=12, droop=r.uniform(0.5, 1.1), cup=0.1, wave=0.0007,
                         wave_freq=4, seed=seed * 70 + k * 3 + lobe, twist=r.uniform(-0.6, 0.6))
        else:
            m = frame(pt, yaw=-yaw, pitch=elev, roll=r.uniform(-0.3, 0.3))
            add_leaf(p, "leaf_javafern", m, col, length=L, width=L * r.uniform(0.15, 0.2), shape=(0.35, 0.85),
                     nu=6, nv=16, droop=r.uniform(0.5, 1.3), cup=0.18, wave=0.0018, wave_freq=4.5,
                     seed=seed * 70 + k, twist=r.uniform(-0.5, 0.5), side_bend=r.uniform(-0.3, 0.3))
    return p


def crypt(seed, leaves=10, size=1.0, flamingo=False):
    """Cryptocoryne wendtii (bronze, crinkled) or 'Flamingo' (pink, smoother)."""
    r = np.random.default_rng(seed)
    p = Part()
    roots(p, r, (0, 0, 0.002), 6, 0.02, "roots_white")
    mat = "leaf_crypt_flamingo" if flamingo else "leaf_crypt_wendtii"
    for k in range(leaves):
        yaw = TAU * k / leaves * 2.618 + r.normal(0, 0.3)
        elev = math.radians(r.uniform(52, 78))
        pl = r.uniform(0.012, 0.03) * size
        d = np.array([math.sin(yaw) * math.cos(elev), math.cos(yaw) * math.cos(elev), math.sin(elev)])
        base = np.array([r.normal(0, 0.003), r.normal(0, 0.003), 0.0])
        stem(p, [base, base + d * pl], 0.0014, 0.0011, "stem_red" if not flamingo else "leaf_crypt_flamingo",
             color=tint(r, 0.9))
        if flamingo:
            L, Wd, wave, freq = r.uniform(0.035, 0.06) * size, 0.27, 0.0007, 3.0
        else:
            L, Wd, wave, freq = r.uniform(0.045, 0.075) * size, 0.24, 0.0022, 7.5
        m = frame(base + d * pl, yaw=-yaw, pitch=elev - math.radians(r.uniform(15, 40)), roll=r.uniform(-0.4, 0.4))
        add_leaf(p, mat, m, tint(r, r.uniform(0.8, 1.08)), length=L, width=L * Wd, shape=(0.45, 0.75), nu=6,
                 nv=14, droop=r.uniform(0.5, 1.1), cup=0.1, wave=wave, wave_freq=freq, seed=seed * 40 + k,
                 base_w=0.15, twist=r.uniform(-0.4, 0.4))
    return p


def grass(seed, leaves=14, size=1.0):
    """Dwarf sagittaria: arching straps (the grassy tufts, front left)."""
    r = np.random.default_rng(seed)
    p = Part()
    roots(p, r, (0, 0, 0.001), 5, 0.02)
    for k in range(leaves):
        yaw = r.uniform(0, TAU)
        elev = math.radians(r.uniform(62, 88))
        L = r.uniform(0.05, 0.12) * size
        m = frame((r.normal(0, 0.003), r.normal(0, 0.003), 0), yaw=yaw, pitch=elev, roll=r.uniform(-0.5, 0.5))
        add_leaf(p, "leaf_grass", m, tint(r, r.uniform(0.8, 1.08)), length=L, width=0.004, shape=(0.06, 0.25),
                 nu=1, nv=10, droop=r.uniform(0.6, 1.6), cup=0.2, seed=seed * 30 + k, twist=r.uniform(-1, 1))
    return p


def stem_bunch(seed, kind, stems=8, height=0.18, size=1.0):
    """Upright stem plants: 'bacopa' (Brazilian moneywort / Bacopa: opposite
    round leaves), 'ludwigia' (narrow red-tipped leaves), 'najas' (guppy
    grass: whorls of needles on thin wandering stems)."""
    r = np.random.default_rng(seed)
    p = Part()
    for s in range(stems):
        h = height * r.uniform(0.65, 1.1)
        b = np.array([r.normal(0, 0.008), r.normal(0, 0.008), 0])
        lean = np.array([r.normal(0, 0.18), r.normal(0, 0.18), 1.0])
        ctrl = [b]
        for t in (0.33, 0.66, 1.0):
            ctrl.append(b + lean * h * t + np.array([r.normal(0, 0.008), r.normal(0, 0.008), 0]) * t * 2)
        if kind == "najas":
            ctrl = [b] + [b + lean * h * t + np.array([r.normal(0, 0.02), r.normal(0, 0.02), 0]) for t in
                          (0.2, 0.4, 0.6, 0.8, 1.0)]
        mat_stem = "stem_red" if kind == "ludwigia" else "stem_green"
        rad = 0.0006 if kind == "najas" else 0.0013
        pts = stem(p, ctrl, rad, rad * 0.7, mat_stem, sides=4, per=6)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        total = seg[-1]
        spacing = {"bacopa": 0.011, "ludwigia": 0.009, "najas": 0.006}[kind]
        n = int(total / spacing)
        rot0 = r.uniform(0, TAU)
        for k in range(1, n + 1):
            t = k / n
            dist = t * total
            i = min(np.searchsorted(seg, dist), len(pts) - 1)
            at = pts[i]
            tan = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
            tan /= np.linalg.norm(tan) + 1e-9
            if kind == "bacopa":
                if t < 0.25 and r.random() < 0.6:
                    if r.random() < 0.4:
                        roots(p, r, at, 1, 0.015)
                    continue
                pair = 2
                if r.random() < 0.08:
                    continue  # a lost leaf pair
                L = 0.0135 * size * (0.55 + 0.45 * min(1, (1 - t) * 4)) * r.uniform(0.8, 1.15)
                elev_deg = r.uniform(15, 40) + 30 * t
                Wr, shape, nu, nv, mat = 0.7, (0.5, 0.5), 4, 6, "leaf_bacopa"
            elif kind == "ludwigia":
                pair = 2
                L = 0.026 * size * (0.55 + 0.45 * min(1, (1 - t) * 3)) * r.uniform(0.85, 1.1)
                elev_deg = 20 + 45 * t
                Wr, shape, nu, nv, mat = 0.32, (0.45, 0.8), 4, 7, "leaf_ludwigia"
            else:
                pair = 3
                L = r.uniform(0.010, 0.018) * size
                elev_deg = r.uniform(20, 55)
                Wr, shape, nu, nv, mat = 0.08, (0.2, 0.6), 1, 3, "leaf_najas"
            rot = rot0 + k * (math.pi / 2 if pair == 2 else 0.9) + r.normal(0, 0.25)
            col = tint(r, 0.85 + 0.25 * t)
            if kind == "ludwigia":
                col = tint(r, 0.9 + 0.2 * t)
            for q in range(pair):
                yaw = rot + TAU * q / pair
                d = np.array([math.sin(yaw), math.cos(yaw), 0.0])
                d = d - tan * np.dot(d, tan)
                d /= np.linalg.norm(d)
                el = math.radians(elev_deg)
                dirv = d * math.cos(el) + tan * math.sin(el)
                m = look_frame(at, dirv, roll=r.uniform(-0.3, 0.3))
                add_leaf(p, mat, m, col, length=L, width=L * Wr, shape=shape, nu=nu, nv=nv,
                         droop=r.uniform(0.1, 0.5), cup=0.25, seed=int(seed * 9000 + s * 300 + k * 3 + q))
            if kind == "najas" and r.random() < 0.08:
                # side branch
                yaw = r.uniform(0, TAU)
                bd = np.array([math.sin(yaw) * 0.6, math.cos(yaw) * 0.6, 0.8])
                bl = r.uniform(0.02, 0.05)
                bp = stem(p, [at, at + bd * bl * 0.5, at + bd * bl], 0.0005, 0.0004, "stem_green", sides=3)
                for kk in range(int(bl / 0.006)):
                    q0 = bp[min(len(bp) - 1, int(kk / (bl / 0.006) * len(bp)))]
                    for q in range(3):
                        yw = kk * 0.9 + TAU * q / 3
                        dirv = np.array([math.sin(yw), math.cos(yw), 0.6])
                        add_leaf(p, "leaf_najas", look_frame(q0, dirv), col, length=r.uniform(0.008, 0.014),
                                 width=0.0012, shape=(0.2, 0.6), nu=1, nv=3, droop=0.3, cup=0,
                                 seed=int(seed * 777 + kk * 3 + q))
    return p


def round_leaf_mesh(radius, seed, segs=18, rings=4, cup=0.18):
    """Pennywort leaf: near-round, scalloped, with a cleft where the petiole
    joins (at the centre: Hydrocotyle leucocephala's leaf is entire with
    a shallow sinus). Normal +Z, planar UV."""
    r = np.random.default_rng(seed)
    verts, uvs, faces = [(0.0, 0.0, 0.0)], [(0.5, 0.5)], []
    lobes = r.integers(7, 10)
    for k in range(1, rings + 1):
        t = k / rings
        for i in range(segs):
            a = TAU * i / segs
            rr = radius * t * (1 + 0.06 * math.sin(lobes * a) * t)
            sinus = 1 - 0.35 * math.exp(-((((a + math.pi) % TAU) - math.pi) / 0.18) ** 2) * t ** 2
            rr *= sinus
            x, y = math.cos(a) * rr, math.sin(a) * rr
            z = cup * radius * t * t + 0.0008 * math.sin(lobes * a) * t
            verts.append((x, y, z))
            uvs.append((0.5 + x / (2 * radius), 0.5 + y / (2 * radius)))
    for i in range(segs):
        faces.append([0, 1 + i, 1 + (i + 1) % segs])
    for k in range(rings - 1):
        a0, a1 = 1 + k * segs, 1 + (k + 1) * segs
        for i in range(segs):
            j = (i + 1) % segs
            faces.append([a0 + i, a1 + i, a1 + j, a0 + j])
    return np.array(verts), faces, uvs


def pennywort_vine(seed, path, leaves_every=0.03):
    """Hydrocotyle leucocephala: a trailing stem along `path` (tank space)
    with round leaves on long petioles turned up toward the light."""
    r = np.random.default_rng(seed)
    p = Part()
    pts = stem(p, path, 0.0015, 0.0011, "stem_green", sides=5, per=8)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    n = int(seg[-1] / leaves_every)
    for k in range(1, n + 1):
        i = min(np.searchsorted(seg, k * leaves_every + r.uniform(-0.006, 0.006)), len(pts) - 1)
        at = pts[i]
        yaw = r.uniform(0, TAU)
        pl = r.uniform(0.02, 0.045)
        d = np.array([math.sin(yaw) * 0.5, math.cos(yaw) * 0.5, 0.85])
        tip = at + d * pl
        if tip[2] > 0.272:  # stay under the surface
            tip[2] = 0.272 - r.uniform(0, 0.004)
        stem(p, [at, (at + tip) / 2 + np.array([0, 0, 0.004]), tip], 0.0008, 0.0006, "stem_green", sides=4, per=3)
        v, f, uv = round_leaf_mesh(r.uniform(0.011, 0.018), seed * 100 + k)
        m = Matrix.Translation(Vector(tip)) @ Matrix.Rotation(r.uniform(0, TAU), 4, "Z") \
            @ Matrix.Rotation(r.uniform(-0.35, 0.35), 4, "X") @ Matrix.Rotation(r.uniform(-0.35, 0.35), 4, "Y")
        q = np.c_[v, np.ones(len(v))] @ np.array(m).T
        p.add(q[:, :3], f, uv, "leaf_pennywort", tint(r, r.uniform(0.85, 1.1)))
        if r.random() < 0.35:
            roots(p, r, at, 2, 0.02)
    return p


def salvinia_node(seed):
    """One Salvinia minima node: a pair of round, upturned floating fronds.
    Floats at z = 0 (the water surface); three.js scatters thousands."""
    r = np.random.default_rng(seed)
    p = Part()
    for side in (-1, 1):
        L = r.uniform(0.008, 0.011)
        m = frame((side * 0.0012, 0, 0.0005), yaw=side * math.radians(r.uniform(70, 100)),
                  pitch=math.radians(r.uniform(4, 14)), roll=0)
        add_leaf(p, "leaf_salvinia", m, tint(r, 1.0, 0.1), length=L, width=L * 0.95, shape=(0.55, 0.55), nu=5,
                 nv=6, droop=-0.25, cup=0.55, seed=seed * 3 + side, base_w=0.2)
    return p
