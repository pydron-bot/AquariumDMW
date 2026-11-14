"""Procedural textures, painted with numpy and saved through Blender.

Arrays are [row, col] with row 0 at the TOP of the image. Leaf and fish
painters work in UV space with v pointing up (v = 1 - row/H), the way the
meshes are unwrapped; three.js loads these with flipY = false, matching
glTF's UV convention.

Normal maps are OpenGL style (+G = +v), what three.js expects.
Everything is seeded: a rebuild gives the same pixels.
"""
import math
from pathlib import Path

import bpy
import numpy as np


# ---------------------------------------------------------------- io

def save(path, img, quality=90):
    """img: uint8 (H, W, 3|4) or float in 0..1. .jpg or .png by extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    h, w, c = a.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :c] = a[:, :, :c] / 255.0
    im = bpy.data.images.new(path.stem, w, h, alpha=(c == 4))
    im.colorspace_settings.name = "Non-Color"  # write bytes as given
    im.pixels.foreach_set(rgba[::-1].ravel())
    im.filepath_raw = str(path.resolve())
    im.file_format = "JPEG" if path.suffix == ".jpg" else "PNG"
    if path.suffix == ".jpg":
        bpy.context.scene.render.image_settings.quality = quality
        im.save(quality=quality)
    else:
        im.save()
    bpy.data.images.remove(im)
    print(f"  tex {path.name} {w}x{h} ({path.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------- building blocks

def rng(seed):
    return np.random.default_rng(seed)


def spectral(h, w, beta=2.0, seed=0, aniso=(1.0, 1.0), lo=1.0):
    """Tileable 1/f^beta noise in [-1, 1]. aniso stretches features:
    (8, 1) penalises vertical frequencies: streaks run vertically."""
    r = rng(seed)
    fy = np.fft.fftfreq(h)[:, None] * h * aniso[0]
    fx = np.fft.fftfreq(w)[None, :] * w * aniso[1]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    amp = 1.0 / np.maximum(f, lo) ** (beta / 2)
    amp[0, 0] = 0
    spec = (r.normal(size=(h, w)) + 1j * r.normal(size=(h, w))) * amp
    n = np.real(np.fft.ifft2(spec))
    n -= n.mean()
    return n / (np.abs(n).max() + 1e-9)


def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = np.asarray(t)[..., None] if np.ndim(t) == 2 else t
    return np.asarray(a, dtype=float) * (1 - t) + np.asarray(b, dtype=float) * t


def normal_map(height, strength=4.0, wrap=True):
    """Height (H, W) -> OpenGL normal map uint8."""
    h = height
    if wrap:
        dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
        dv = (np.roll(h, 1, 0) - np.roll(h, -1, 0)) * 0.5  # row-1 is +v
    else:
        dx = np.gradient(h, axis=1)
        dv = -np.gradient(h, axis=0)
    n = np.dstack([-dx * strength, -dv * strength, np.ones_like(h)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n * 0.5 + 0.5) * 255 + 0.5).astype(np.uint8)


def stamp_disks(height, color, xs, ys, rs, cols, hscale=1.0, base=None, wrap=True):
    """Dome-shaped grains: raises `height` and paints `color` where the new
    grain is on top. cols are 0-255 rgb rows."""
    H, W = height.shape
    for k in range(len(xs)):
        r = rs[k]
        R = int(math.ceil(r)) + 1
        cx, cy = xs[k], ys[k]
        ix, iy = int(cx), int(cy)
        cols_i = np.arange(ix - R, ix + R + 1)
        rows_i = np.arange(iy - R, iy + R + 1)
        if wrap:
            ci, ri = cols_i % W, rows_i % H
        else:
            ok_c = (cols_i >= 0) & (cols_i < W)
            ok_r = (rows_i >= 0) & (rows_i < H)
            cols_i, rows_i = cols_i[ok_c], rows_i[ok_r]
            ci, ri = cols_i, rows_i
            if len(ci) == 0 or len(ri) == 0:
                continue
        dx = cols_i[None, :] + 0.5 - cx
        dy = rows_i[:, None] + 0.5 - cy
        d2 = (dx * dx + dy * dy) / (r * r)
        dome = np.sqrt(np.clip(1 - d2, 0, 1))
        b = 0.0 if base is None else base[k]
        hnew = b + dome * r * hscale
        sub_h = height[np.ix_(ri, ci)]
        top = (dome > 0) & (hnew > sub_h)
        if not top.any():
            continue
        sub_h[top] = hnew[top]
        height[np.ix_(ri, ci)] = sub_h
        sub_c = color[np.ix_(ri, ci)]
        # light from upper left, so grains read as rounded
        shade = 0.72 + 0.28 * np.clip(dome - (dx + dy) / (r * 2.8), 0, 1.2)
        sub_c[top] = (np.asarray(cols[k], dtype=float)[None, :] * shade[top][:, None])
        color[np.ix_(ri, ci)] = sub_c


def grains(size, count, rmin, rmax, palette, weights, seed, ground, hscale=1.0):
    r = rng(seed)
    height = np.zeros((size, size))
    color = np.empty((size, size, 3))
    color[:] = ground
    xs = r.uniform(0, size, count)
    ys = r.uniform(0, size, count)
    rs = r.uniform(rmin, rmax, count) ** 1.0
    pal = np.asarray(palette, dtype=float)
    pick = r.choice(len(pal), count, p=np.asarray(weights) / np.sum(weights))
    jitter = r.normal(1.0, 0.09, (count, 1))
    cols = np.clip(pal[pick] * jitter, 0, 255)
    base = r.uniform(0, rmax * 0.4, count)
    stamp_disks(height, color, xs, ys, rs, cols, hscale, base)
    return color, height


# ---------------------------------------------------------------- tileable surfaces

def sand(out, size=1024):
    pal = [(196, 168, 122), (222, 204, 168), (160, 120, 78), (118, 84, 54), (70, 54, 42),
           (238, 232, 220), (150, 96, 58), (140, 134, 124), (182, 150, 104)]
    w = [22, 12, 14, 9, 5, 6, 5, 6, 18]
    color, height = grains(size, 34000, 2.6, 6.5, pal, w, 11, (92, 72, 52))
    color *= (1 + 0.1 * spectral(size, size, 2.2, 12))[..., None]
    save(out / "sand_color.jpg", np.clip(color, 0, 255).astype(np.uint8))
    save(out / "sand_normal.png", normal_map(height, 0.9))


def soil(out, size=512):
    """Aquasoil showing at the back and through the front glass."""
    pal = [(48, 36, 28), (64, 48, 36), (34, 27, 22), (82, 62, 44), (24, 20, 18)]
    color, height = grains(size, 5200, 4.0, 8.5, pal, [30, 20, 25, 8, 12], 21, (18, 14, 12))
    save(out / "soil_color.jpg", np.clip(color, 0, 255).astype(np.uint8))
    save(out / "soil_normal.png", normal_map(height, 0.8))


def bark(out, w=512, h=1024):
    """Driftwood: grain runs along v (the branch). Dark red-brown, deep
    cracks, a little green film and the white specks seen on the real wood."""
    r = rng(31)
    streak = spectral(h, w, 2.4, 32, aniso=(9.0, 1.0))
    fine = spectral(h, w, 1.4, 33, aniso=(5.0, 1.0))
    blotch = spectral(h, w, 3.0, 34)
    gap = smooth(0.86, 0.98, 1 - np.abs(spectral(h, w, 2.2, 36, aniso=(10.0, 1.0))))
    hgt = 0.55 * streak + 0.25 * fine + 0.25 * blotch - 1.2 * gap
    dark = np.array([34, 22, 16.0])
    mid = np.array([70, 44, 30.0])
    light = np.array([112, 76, 52.0])
    t = np.clip(0.5 + 0.5 * (0.7 * streak + 0.3 * fine), 0, 1)
    col = mix(dark, mid, smooth(0.15, 0.6, t))
    col = mix(col, light, smooth(0.65, 0.95, t) * 0.6)
    col = mix(col, np.array([22, 14, 10.0]), gap * 0.9)
    film = smooth(0.35, 0.8, blotch) * 0.35
    col = mix(col, np.array([70, 74, 40.0]), film)
    # tiny white specks (ostracods / snail eggs on the real wood)
    spk = np.zeros((h, w))
    n = 900
    xs, ys = r.uniform(0, w, n), r.uniform(0, h, n)
    for x, y in zip(xs.astype(int), ys.astype(int)):
        spk[y % h, x % w] = 1
        spk[(y + 1) % h, x % w] = 0.7
        spk[y % h, (x + 1) % w] = 0.7
    col = mix(col, np.array([226, 222, 200.0]), spk * 0.85)
    save(out / "bark_color.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / "bark_normal.png", normal_map(hgt * 6.0, 2.2))


def rock(out, size=512):
    """Dark porous rock (the black lumps under the wood)."""
    r = rng(41)
    base = spectral(size, size, 2.2, 42)
    col = mix(np.array([30, 28, 27.0]), np.array([68, 62, 56.0]), smooth(-0.6, 0.8, base))
    hgt = base * 2.0
    n = 1400
    for _ in range(n):
        x, y = r.uniform(0, size, 2)
        rad = r.uniform(1.5, 7) ** 1.0
        R = int(rad) + 2
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        d = np.sqrt((xx + 0.5 - (x % 1)) ** 2 + (yy + 0.5 - (y % 1)) ** 2) / rad
        pit = np.clip(1 - d, 0, 1)
        ri = (np.arange(int(y) - R, int(y) + R + 1) % size)
        ci = (np.arange(int(x) - R, int(x) + R + 1) % size)
        hgt[np.ix_(ri, ci)] -= pit * 1.4
        sub = col[np.ix_(ri, ci)]
        sub *= (1 - pit[..., None] * 0.65)
        col[np.ix_(ri, ci)] = sub
    save(out / "rock_color.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / "rock_normal.png", normal_map(hgt, 3.0))


def pebble(out, size=512):
    """Smooth river stone: speckled, used with per-stone tints."""
    r = rng(51)
    base = spectral(size, size, 2.6, 52)
    col = mix(np.array([150, 146, 136.0]), np.array([196, 190, 176.0]), smooth(-0.7, 0.7, base))
    n = 6000
    xs, ys = r.uniform(0, size, n).astype(int), r.uniform(0, size, n).astype(int)
    tone = r.choice([40, 70, 230], n, p=[0.5, 0.35, 0.15])
    for x, y, t in zip(xs, ys, tone):
        col[y, x] = col[y, x] * 0.3 + t * 0.7
    blur = (col + np.roll(col, 1, 0) + np.roll(col, 1, 1)) / 3
    hgt = spectral(size, size, 3.0, 53) * 0.4
    save(out / "pebble_color.jpg", np.clip(blur, 0, 255).astype(np.uint8))
    save(out / "pebble_normal.png", normal_map(hgt, 1.0))


def condensation(out, size=512):
    """Drops on the underside of the glass lid: normal map (rgb) and drop
    coverage (alpha)."""
    r = rng(61)
    height = np.zeros((size, size))
    color = np.zeros((size, size, 3))
    n = 2600
    xs, ys = r.uniform(0, size, n), r.uniform(0, size, n)
    rs = np.clip(r.lognormal(1.2, 0.55, n), 1.2, 22)
    stamp_disks(height, color, xs, ys, rs, np.ones((n, 3)) * 255, 0.5)
    nm = normal_map(height, 1.6)
    cov = smooth(0.0, 0.4, height)
    rgba = np.dstack([nm, (cov * 255).astype(np.uint8)])
    save(out / "drops.png", rgba)


def sponge(out, size=256):
    r = rng(71)
    base = spectral(size, size, 1.2, 72)
    holes = smooth(0.25, 0.6, spectral(size, size, 1.6, 73))
    col = mix(np.array([36, 38, 36.0]), np.array([8, 9, 9.0]), holes)
    col = mix(col, np.array([70, 64, 44.0]), smooth(0.4, 0.9, base) * 0.3)
    save(out / "sponge_color.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / "sponge_normal.png", normal_map(-holes * 3, 1.5))


def desk(out, w=1024, h=512):
    """Dark grey woodgrain laminate of the stand."""
    s = spectral(h, w, 2.8, 81, aniso=(1.0, 14.0))
    f = spectral(h, w, 1.6, 82, aniso=(1.0, 9.0))
    t = 0.5 + 0.5 * (0.7 * s + 0.3 * f)
    col = mix(np.array([30, 29, 30.0]), np.array([58, 56, 57.0]), smooth(0.2, 0.9, t))
    save(out / "desk_color.jpg", np.clip(col, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------- leaves

def leaf_uv(w, h):
    v = 1 - (np.arange(h)[:, None] + 0.5) / h
    u = (np.arange(w)[None, :] + 0.5) / w
    return np.broadcast_to(u, (h, w)), np.broadcast_to(v, (h, w))


def leaf(out, name, base, tip, rib, edge, veins=10, slope=0.35, arc=1.4, vein_dark=0.0,
         vein_light=0.0, rib_w=0.018, spots=None, bands=None, net=0.0, seed=0, size=(256, 512)):
    """A leaf painted in its own UV space: u across (midrib at 0.5), v from
    base (0) to tip (1). Lateral veins leave the midrib and sweep tipward."""
    W, H = size
    u, v = leaf_uv(W, H)
    s = np.abs(u - 0.5) * 2
    col = mix(np.array(base, float), np.array(tip, float), smooth(0.0, 1.0, v))
    noise = spectral(H, W, 2.4, seed + 1)
    col *= (1 + 0.08 * noise)[..., None]
    phase = (v - slope * s ** arc) * veins
    vd = np.abs(phase - np.round(phase))
    vein = (1 - smooth(0.0, 0.08, vd)) * smooth(0.02, 0.12, s) * (1 - smooth(0.85, 1.0, s))
    height = -vein * 0.6
    if net:
        cross = spectral(H, W, 1.0, seed + 2)
        cell = 1 - smooth(0.0, 0.07, np.abs(cross))
        vein = np.maximum(vein, cell * net * smooth(0.05, 0.2, s))
        height -= cell * net * 0.4
    if vein_dark:
        col = mix(col, col * 0.55, vein * vein_dark)
    if vein_light:
        col = mix(col, np.minimum(col * 1.35 + 20, 255), vein * vein_light)
    ribm = np.exp(-((u - 0.5) / rib_w) ** 2)
    col = mix(col, np.array(rib, float), ribm * 0.85)
    height += ribm * 0.8
    if bands:  # crypt wendtii's darker cross-banding
        bcol, freq, amt = bands
        b = 0.5 + 0.5 * np.sin(v * freq * 2 * math.pi + spectral(H, W, 3.0, seed + 3) * 3)
        col = mix(col, np.array(bcol, float), smooth(0.55, 0.95, b) * amt)
    col = mix(col, np.array(edge, float), smooth(0.86, 1.0, s) * 0.6)
    if spots:
        scol, n, rad = spots
        r = rng(seed + 4)
        for _ in range(n):
            cu, cv = r.uniform(0.15, 0.85), r.uniform(0.1, 0.9)
            rr = r.uniform(0.4, 1.0) * rad
            d = np.sqrt(((u - cu) * W) ** 2 + ((v - cv) * H) ** 2)
            m = 1 - smooth(rr * 0.6, rr, d)
            col = mix(col, np.array(scol, float), m * 0.8)
    save(out / f"leaf_{name}.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / f"leaf_{name}_n.png", normal_map(height * 3, 1.6, wrap=False))


def round_leaf(out, name, base, edge, rib, veins=9, seed=0, size=256):
    """Peltate/round leaf (pennywort): veins radiate from the petiole at the
    centre. UV is planar, centre (0.5, 0.5)."""
    u, v = leaf_uv(size, size)
    x, y = (u - 0.5) * 2, (v - 0.5) * 2
    rad = np.sqrt(x * x + y * y)
    ang = np.arctan2(y, x)
    col = mix(np.array(base, float), np.array(edge, float), smooth(0.5, 1.0, rad))
    col *= (1 + 0.07 * spectral(size, size, 2.2, seed))[..., None]
    a = (ang / (2 * math.pi) + 0.5) * veins
    vd = np.abs(a - np.round(a)) / np.maximum(rad, 0.05)
    vein = (1 - smooth(0.0, 0.07, vd * 0.4)) * smooth(0.05, 0.2, rad)
    col = mix(col, np.array(rib, float), vein * 0.6)
    save(out / f"leaf_{name}.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / f"leaf_{name}_n.png", normal_map(vein * 1.2, 1.5, wrap=False))


def salvinia_leaf(out, size=256):
    """Salvinia minima frond: velvety, rows of hair-tipped papillae."""
    u, v = leaf_uv(size, size)
    col = mix(np.array([118, 150, 58.0]), np.array([168, 186, 92.0]), smooth(0.0, 1.0, v))
    col *= (1 + 0.06 * spectral(size, size, 2.0, 91))[..., None]
    gu, gv = u * 18, v * 22 + 0.5 * (np.floor(u * 18) % 2)
    d = np.sqrt((gu - np.round(gu)) ** 2 + (gv - np.round(gv)) ** 2)
    pap = 1 - smooth(0.12, 0.32, d)
    col = mix(col, np.array([214, 222, 170.0]), pap * 0.55)
    rib = np.exp(-((u - 0.5) / 0.03) ** 2)
    col = mix(col, np.array([110, 132, 50.0]), rib * 0.5)
    save(out / "leaf_salvinia.jpg", np.clip(col, 0, 255).astype(np.uint8))
    save(out / "leaf_salvinia_n.png", normal_map(pap * 1.5 - rib, 1.6, wrap=False))


def salvinia_roots(out, w=256, h=512, seed=101):
    """Card of the hanging, hairy submerged fronds: golden brown filaments
    that branch into fine hairs. RGBA; top of the card = water surface."""
    r = rng(seed)
    col = np.zeros((h, w, 3))
    alpha = np.zeros((h, w))
    for strand in range(26):
        x = r.uniform(10, w - 10)
        length = r.uniform(0.35, 0.98) * h
        sway = r.uniform(-0.3, 0.3)
        t = np.linspace(0, 1, int(length))
        xs = x + np.cumsum(r.normal(0, 0.35, len(t))) + sway * t * 60
        ys = t * length
        tone = r.uniform(0.75, 1.15)
        c = np.array([176, 136, 72.0]) * tone
        for i in range(0, len(t)):
            yi, xi = int(ys[i]), int(xs[i]) % w
            if yi >= h:
                break
            wid = 1.4 * (1 - t[i] * 0.6)
            for dx in range(-2, 3):
                a = max(0.0, 1 - abs(dx) / (wid + 0.5))
                if a <= 0:
                    continue
                cc = (xi + dx) % w
                alpha[yi, cc] = max(alpha[yi, cc], a)
                col[yi, cc] = c
            # side hairs
            if r.random() < 0.55 and t[i] > 0.03:
                ang = r.uniform(0.25, 1.2) * (1 if r.random() < 0.5 else -1)
                hl = r.uniform(6, 22) * (1 - 0.5 * t[i])
                for k in range(int(hl)):
                    hx = int(xs[i] + math.sin(ang) * k) % w
                    hy = int(ys[i] + math.cos(abs(ang)) * k * 0.9)
                    if 0 <= hy < h:
                        a = 0.55 * (1 - k / hl)
                        if a > alpha[hy, hx]:
                            alpha[hy, hx] = a
                            col[hy, hx] = c * 1.1
    # soften
    a2 = (alpha + np.roll(alpha, 1, 1) * 0.5 + np.roll(alpha, -1, 1) * 0.5) / 2
    alpha = np.maximum(alpha, a2 * 0.8)
    col = np.where(alpha[..., None] > 0, col, np.array([140, 105, 60.0]))
    rgba = np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
    save(out / "salvinia_roots.png", rgba)


def plain(out, name, rgb, var=0.06, seed=0, size=128):
    col = np.empty((size, size, 3))
    col[:] = rgb
    col *= (1 + var * spectral(size, size, 2.0, seed))[..., None]
    save(out / f"{name}.jpg", np.clip(col, 0, 255).astype(np.uint8))


def all_leaves(out):
    leaf(out, "anubias", base=(30, 62, 26), tip=(46, 82, 34), rib=(70, 104, 50), edge=(26, 52, 22),
         veins=14, slope=0.55, arc=1.5, vein_light=0.25, rib_w=0.02, seed=200,
         spots=((88, 96, 40), 6, 3.0))
    leaf(out, "javafern", base=(58, 104, 40), tip=(88, 132, 52), rib=(46, 82, 30), edge=(52, 92, 36),
         veins=16, slope=0.6, arc=1.2, vein_dark=0.5, net=0.6, rib_w=0.022, seed=210,
         spots=((30, 24, 18), 7, 4.0))
    leaf(out, "crypt_wendtii", base=(82, 70, 40), tip=(104, 84, 50), rib=(130, 92, 66),
         edge=(92, 60, 48), veins=7, slope=0.9, arc=1.1, vein_dark=0.35, rib_w=0.02, seed=220,
         bands=((54, 40, 30), 9, 0.55))
    leaf(out, "crypt_flamingo", base=(206, 132, 128), tip=(232, 160, 150), rib=(240, 190, 180),
         edge=(190, 110, 110), veins=7, slope=0.9, arc=1.1, vein_light=0.3, rib_w=0.02, seed=230)
    leaf(out, "bacopa", base=(96, 150, 50), tip=(130, 178, 64), rib=(120, 170, 70), edge=(110, 160, 56),
         veins=5, slope=0.5, vein_light=0.2, rib_w=0.03, seed=240)
    leaf(out, "ludwigia", base=(80, 120, 44), tip=(176, 76, 52), rib=(150, 90, 60), edge=(140, 70, 50),
         veins=8, slope=0.6, vein_dark=0.2, rib_w=0.025, seed=250)
    leaf(out, "grass", base=(70, 120, 46), tip=(110, 158, 66), rib=(90, 140, 60), edge=(80, 130, 52),
         veins=0, rib_w=0.06, seed=260, size=(64, 512))
    leaf(out, "najas", base=(84, 118, 42), tip=(118, 140, 56), rib=(84, 118, 42), edge=(84, 118, 42),
         veins=0, rib_w=0.1, seed=270, size=(32, 128))
    round_leaf(out, "pennywort", base=(112, 168, 54), edge=(92, 146, 46), rib=(160, 200, 98), seed=280)
    salvinia_leaf(out)
    plain(out, "stem_green", (80, 120, 50), seed=290)
    plain(out, "stem_red", (122, 60, 44), seed=291)
    plain(out, "rhizome", (58, 72, 38), seed=292)
    plain(out, "roots_white", (150, 136, 104), seed=293)
