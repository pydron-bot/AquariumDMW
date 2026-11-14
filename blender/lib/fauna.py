"""White Cloud Mountain Minnow (Tanichthys albonubes) and ramshorn snail.

The minnow is ~3.6 cm, nose toward Blender -Y (three.js +Z, so lookAt
aims it), body origin at its centre. Its UVs are a side projection over
the whole silhouette (fins included), so the side-view painting in
fish_texture() lands directly; both flanks share it. Fins are thin planes
in the body's mid plane (pectorals and pelvics angled out), drawn in a
separate alpha-blended material.
"""
import math

import numpy as np

from . import textures as T
from .util import Part, grid_faces

L = 0.036          # total length incl. tail
TAU = 2 * math.pi

# Profile along s (0 = snout, 1 = tail tip); body ends at the peduncle.
BODY_END = 0.80


def top(s):
    s = np.asarray(s, float)
    return 0.105 * np.sin(np.clip(s / BODY_END, 0, 1) * math.pi) ** 0.75 * (1 - 0.35 * np.clip(s / BODY_END, 0, 1)) + 0.018


def bottom(s):
    s = np.asarray(s, float)
    return -(0.085 * np.sin(np.clip(s / BODY_END, 0, 1) * math.pi) ** 0.7 * (1 - 0.25 * np.clip(s / BODY_END, 0, 1)) + 0.018)


def half_width(s):
    s = np.asarray(s, float)
    return 0.05 * np.sin(np.clip(s / BODY_END, 0, 1) * math.pi) ** 0.9 * (1 - 0.5 * np.clip(s / BODY_END, 0, 1)) + 0.006


# Silhouette bounds (in units of L) for the side projection
ZMIN, ZMAX = -0.17, 0.2


def side_uv(s, z):
    return (s, (z / L - ZMIN) / (ZMAX - ZMIN))


def fish():
    body, fins = Part(), Part()
    ns, nr = 30, 16
    verts, uvs = [], []
    for j in range(ns + 1):
        s = j / ns * BODY_END
        t, b, w = top(s) * L, bottom(s) * L, half_width(s) * L
        # snout: slightly upturned mouth
        zc = (t + b) / 2 + (0.006 * L if s < 0.06 else 0)
        hh = (t - b) / 2
        for i in range(nr + 1):
            a = TAU * i / nr
            ca, sa = math.cos(a), math.sin(a)
            # superellipse cross-section, flat-sided like the real fish
            x = w * np.sign(ca) * abs(ca) ** 0.8
            z = zc + hh * np.sign(sa) * abs(sa) ** 0.9
            y = -L / 2 + s * L
            verts.append((x, y, z))
            uvs.append(side_uv(s, z))
    faces = grid_faces(nr, ns)
    # nose cap
    c = len(verts)
    verts.append((0, -L / 2 - 0.0004, (top(0) + bottom(0)) * L / 2))
    uvs.append(side_uv(0, verts[-1][2]))
    for i in range(nr):
        faces.append([i + 1, i, c])
    body.add(np.array(verts), faces, uvs, "fish_body")

    def fin(outline, offset_x=0.0, tilt=0.0, mat="fish_fin"):
        """outline: [(s, z_over_L)] polygon in the side plane; fanned from
        its first point. tilt rotates it out of the plane about the fin's
        base (pectorals/pelvics)."""
        pts = []
        for s, zl in outline:
            pts.append(np.array([0.0, -L / 2 + s * L, zl * L]))
        base = pts[0]
        vs, uv = [], []
        for q in pts:
            d = q - base
            r = np.array([d[0] + math.sin(tilt) * abs(d[2] + d[1] * 0.3), d[1], d[2] * math.cos(tilt)])
            vs.append(base + r + np.array([offset_x, 0, 0]))
            uv.append(side_uv((q[1] + L / 2) / L, q[2]))
        f = [[0, i, i + 1] for i in range(1, len(vs) - 1)]
        fins.add(np.array(vs), f, uv, mat)

    def fin_strip(base_pts, tip_pts, offset_x=0.0, tilt=0.0):
        """A fin as a strip between its base line and outer edge, so the
        vertex wave bends it smoothly."""
        n = len(base_pts)
        vs, uv = [], []
        for k in range(n):
            for (s, zl) in (base_pts[k], tip_pts[k]):
                vs.append(np.array([offset_x, -L / 2 + s * L, zl * L]))
                uv.append(side_uv(s, zl * L))
        for k in range(n):
            b, t = vs[2 * k], vs[2 * k + 1]
            d = t - b
            vs[2 * k + 1] = b + np.array([math.sin(tilt) * np.linalg.norm(d) * np.sign(offset_x or 1), d[1],
                                          d[2] * math.cos(tilt)])
        f = []
        for k in range(n - 1):
            a = 2 * k
            f.append([a, a + 2, a + 3, a + 1])
        fins.add(np.array(vs), f, uv, "fish_fin")

    # caudal: forked, ~20% of length
    cb = [(0.78, -0.03 + 0.0 * k) for k in range(1)]
    tail_base = [(0.775, z) for z in np.linspace(-0.035, 0.045, 7)]
    tail_tip = []
    for k, (s, z) in enumerate(tail_base):
        t = k / 6
        spread = -0.12 + t * 0.27
        fork = 0.17 - 0.06 * math.sin(t * math.pi) ** 0.6
        tail_tip.append((s + fork + 0.03 * (1 - math.sin(t * math.pi)), spread))
    fin_strip(tail_base, tail_tip)
    # dorsal: mid-back, tall in males
    dbase = [(s, float(top(s)) - 0.01) for s in np.linspace(0.47, 0.66, 6)]
    dtip = [(s + 0.05 + 0.04 * k / 5, float(top(s)) + 0.075 - 0.045 * k / 5) for k, s in enumerate(np.linspace(0.47, 0.66, 6))]
    fin_strip(dbase, dtip)
    # anal: long, under the rear half
    abase = [(s, float(bottom(s)) + 0.008) for s in np.linspace(0.5, 0.72, 6)]
    atip = [(s + 0.045 + 0.03 * k / 5, float(bottom(s)) - 0.065 + 0.04 * k / 5) for k, s in enumerate(np.linspace(0.5, 0.72, 6))]
    fin_strip(abase, atip)
    # pelvics and pectorals, angled out to each side
    for sx in (-1, 1):
        x0 = sx * float(half_width(0.42)) * L * 0.8
        fin_strip([(0.4, -0.06), (0.44, -0.065)], [(0.47, -0.105), (0.49, -0.1)], offset_x=x0, tilt=0.6 * sx)
        x1 = sx * float(half_width(0.2)) * L * 0.9
        fin_strip([(0.2, -0.02), (0.23, -0.035)], [(0.3, -0.04), (0.31, -0.07)], offset_x=x1, tilt=0.9 * sx)

    # eyes
    eye = Part()
    for sx in (-1, 1):
        cx, cy, cz = sx * float(half_width(0.12)) * L * 0.75, -L / 2 + 0.12 * L, 0.03 * L
        rad = 0.032 * L
        vs, uv = [], []
        n1, n2 = 10, 6
        for j in range(n2 + 1):
            th = math.pi * j / n2
            for i in range(n1 + 1):
                ph = TAU * i / n1
                d = np.array([math.cos(th) * sx, math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)])
                vs.append((cx + d[0] * rad * 0.6, cy + d[1] * rad, cz + d[2] * rad))
                uv.append((0.5 + 0.5 * math.sin(th) * math.cos(ph), 0.5 + 0.5 * math.sin(th) * math.sin(ph)))
        eye.add(np.array(vs), grid_faces(n1, n2), uv, "fish_eye")
    return body, fins, eye


def fish_texture(out, W=512, H=256):
    """Side view painting in the fish's UV space (u = s, v over ZMIN..ZMAX).
    Olive-bronze back, a gold-white iridescent stripe from eye to tail,
    a dark band under it, a silvery belly; red caudal with a dark spot at
    its base, red dorsal and anal fins edged in white."""
    u, v = T.leaf_uv(W, H)
    s = u
    z = (ZMIN + v * (ZMAX - ZMIN))       # in units of L
    t, b = top(s), bottom(s)
    rel = np.clip((z - b) / np.maximum(t - b, 1e-3), 0, 1)  # 0 belly .. 1 back
    back = np.array([70, 58, 40.0])
    belly = np.array([176, 168, 150.0])
    col = T.mix(belly, back, T.smooth(0.2, 0.75, rel))
    # lateral stripe: dark band under a bright gold line
    stripe_c = 0.52 + 0.04 * (s - 0.5)
    band = np.exp(-((rel - (stripe_c - 0.12)) / 0.09) ** 2) * T.smooth(0.1, 0.25, s)
    col = T.mix(col, np.array([58, 44, 60.0]), band * 0.75)
    gold = np.exp(-((rel - stripe_c) / 0.045) ** 2) * T.smooth(0.12, 0.3, s)
    col = T.mix(col, np.array([238, 222, 160.0]), gold * 0.9)
    # scales
    sc = 0.5 + 0.5 * np.sin(s * 160) * np.sin(rel * 40 + s * 80)
    col *= (0.93 + 0.07 * sc)[..., None]
    # gill cover + head
    head = T.smooth(0.2, 0.12, s)
    col = T.mix(col, np.array([150, 130, 100.0]), head * 0.4 * T.smooth(0.3, 0.6, rel))
    # caudal peduncle spot and red tail
    fin_zone = s > BODY_END - 0.02
    red = np.array([206, 52, 40.0])
    tail = T.smooth(0.76, 0.8, s)
    tail_col = T.mix(red, np.array([240, 230, 210.0]), T.smooth(0.9, 0.98, s) * 0.8)
    spot = np.exp(-(((s - 0.81) / 0.025) ** 2 + ((z - 0.005) / 0.04) ** 2))
    tail_col = T.mix(tail_col, np.array([40, 30, 30.0]), spot * 0.8)
    out_body = (z > t + 0.004) | (z < b - 0.004)
    # dorsal and anal: red with a white edge
    edge_d = np.clip((z - t) / 0.07, 0, 1)
    edge_a = np.clip((b - z) / 0.06, 0, 1)
    fin_col = T.mix(red * 0.9, np.array([246, 240, 220.0]), T.smooth(0.65, 0.9, np.maximum(edge_d, edge_a)))
    fin_col = T.mix(np.array([170, 120, 90.0]), fin_col, T.smooth(0.0, 0.25, np.maximum(edge_d, edge_a)))
    # pelvic/pectoral area (small, low): pale with red tips
    col = np.where((out_body & (s < BODY_END - 0.02))[..., None], fin_col, col)
    col = np.where(tail[..., None] > 0.5, tail_col, col)
    alpha = np.ones((H, W))
    rays = 0.75 + 0.25 * np.abs(np.sin((z * 140) + s * 30))
    alpha = np.where(out_body | (s > BODY_END), 0.62 * rays, alpha)
    alpha = np.where(tail > 0.5, (0.78 - 0.35 * T.smooth(0.9, 1.0, s)) * rays, alpha)
    rgba = np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
    T.save(out / "fish.png", rgba)
    # eye: gold iris, black pupil
    eu, ev = T.leaf_uv(64, 64)
    er = np.sqrt((eu - 0.5) ** 2 + (ev - 0.5) ** 2) * 2
    ec = T.mix(np.array([10, 10, 12.0]), np.array([200, 170, 90.0]), T.smooth(0.45, 0.55, er))
    ec = T.mix(ec, np.array([60, 50, 40.0]), T.smooth(0.85, 1.0, er))
    T.save(out / "fish_eye.jpg", np.clip(ec, 0, 255).astype(np.uint8))


def ramshorn():
    """Planorbid snail: flat coil, ~9 mm, body peeking out. Origin at the
    foot, shell up; three.js sticks it to glass and leaves."""
    p = Part()
    turns = 3.2
    n = 90
    sides = 10
    verts, uvs = [], []
    for j in range(n + 1):
        t = j / n
        ang = t * turns * TAU
        r0 = 0.0005 + 0.0038 * (t ** 1.4)
        tube_r = 0.0004 + 0.0018 * (t ** 1.3)
        cx, cz = math.cos(ang) * r0, math.sin(ang) * r0
        for i in range(sides + 1):
            a = TAU * i / sides
            # tube around the coil, coil in the XZ plane (standing on edge)
            dx = math.cos(ang) * math.cos(a) * tube_r
            dz = math.sin(ang) * math.cos(a) * tube_r
            dy = math.sin(a) * tube_r * 0.9
            verts.append((cx + dx, dy, cz + dz + 0.0045))
            uvs.append((i / sides, t * turns))
    faces = grid_faces(sides, n)
    p.add(np.array(verts), faces, uvs, "snail_shell")
    # foot
    fv, fu = [], []
    for j in range(6):
        for i in range(9):
            a = TAU * i / 8
            rr = 0.0022 * math.sin(math.pi * (j + 0.5) / 6)
            fv.append((math.cos(a) * rr * 1.6 + (j - 2.5) * 0.0011, math.sin(a) * rr, 0.0004 + 0.0004 * math.sin(a)))
            fu.append((i / 8, j / 5))
    p.add(np.array(fv), grid_faces(8, 5), fu, "snail_body")
    return p


def snail_texture(out, size=256):
    u, v = T.leaf_uv(size, size)
    bands = 0.5 + 0.5 * np.sin(v * 2 * math.pi * 14 + T.spectral(size, size, 2.0, 501) * 2)
    col = T.mix(np.array([120, 58, 36.0]), np.array([164, 92, 54.0]), bands)
    col *= (1 + 0.1 * T.spectral(size, size, 2.0, 502))[..., None]
    T.save(out / "snail_shell.jpg", np.clip(col, 0, 255).astype(np.uint8))
