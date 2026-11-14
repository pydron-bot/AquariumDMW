"""Mesh, noise and image helpers shared by the generators.

Conventions: 1 unit = 1 m. Blender Z is up; the glTF exporter turns that into
three.js +Y, and Blender -Y into three.js +Z (the tank's front glass).
Tank space: origin at the centre of the tank floor (inside the glass).
"""
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

OUT = Path("public/assets")


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


# ---------------------------------------------------------------- materials

def material(name):
    """A named placeholder material: three.js swaps in its own by name."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    return mat


# ---------------------------------------------------------------- mesh parts

class Part:
    """Geometry being assembled: verts (N,3), faces (lists), uvs per vertex,
    one material name per face. Parts merge, then become one Blender mesh."""

    def __init__(self):
        self.v = []
        self.uv = []
        self.f = []
        self.fm = []
        self.col = []

    def add(self, verts, faces, uvs, mat, color=(1, 1, 1)):
        base = len(self.v)
        self.v.extend(map(tuple, verts))
        self.uv.extend(map(tuple, uvs))
        self.col.extend([tuple(color)] * len(verts))
        for f in faces:
            self.f.append([base + i for i in f])
            self.fm.append(mat)
        return self

    def merge(self, other, matrix=None):
        base = len(self.v)
        if matrix is None:
            self.v.extend(other.v)
        else:
            m = np.array(matrix)
            p = np.c_[np.array(other.v), np.ones(len(other.v))] @ m.T
            self.v.extend(map(tuple, p[:, :3]))
        self.uv.extend(other.uv)
        self.col.extend(other.col)
        for f, mname in zip(other.f, other.fm):
            self.f.append([base + i for i in f])
            self.fm.append(mname)
        return self

    def transformed(self, matrix):
        return Part().merge(self, matrix)

    def build(self, name, collection=None, smooth=True):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(self.v, [], self.f)
        mats = []
        for mname in self.fm:
            if mname not in mats:
                mats.append(mname)
        for mname in mats:
            mesh.materials.append(material(mname))
        idx = {m: i for i, m in enumerate(mats)}
        mesh.polygons.foreach_set("material_index", [idx[m] for m in self.fm])
        uvl = mesh.uv_layers.new(name="UVMap")
        loop_v = np.zeros(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loop_v)
        uv = np.array(self.uv, dtype=np.float32)
        uvl.data.foreach_set("uv", uv[loop_v].ravel())
        # Per-vertex tint (leaf age, wood tone): three.js reads it as `color`.
        ca = mesh.color_attributes.new("Col", "BYTE_COLOR", "POINT")
        col = np.array(self.col, dtype=np.float32)
        col = np.c_[col, np.ones(len(col))]
        ca.data.foreach_set("color", col.ravel())
        mesh.update()
        if smooth:
            mesh.shade_smooth()
        obj = bpy.data.objects.new(name, mesh)
        (collection or bpy.context.scene.collection).objects.link(obj)
        return obj


def grid_faces(cols, rows, wrap=False):
    """Quads for a (rows+1) x (cols+1) vertex grid, row-major."""
    w = cols + 1
    faces = []
    for j in range(rows):
        for i in range(cols):
            a = j * w + i
            faces.append([a, a + 1, a + 1 + w, a + w])
    return faces


def tube(points, radii, sides=8, uv_scale=1.0, bump=None, cap=True):
    """A tube along a polyline (parallel-transport frames).
    UV: u around (0..1, seam duplicated), v along in units of the mean
    circumference, so bark/stem textures keep their aspect.
    bump(p, normal, i, k) -> radius multiplier, for knots and gnarl."""
    P = np.asarray(points, dtype=float)
    R = np.asarray(radii, dtype=float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    ref = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
    N0 = np.cross(T[0], ref)
    N0 /= np.linalg.norm(N0)
    Ns = [N0]
    for i in range(1, n):
        b = np.cross(T[i - 1], T[i])
        nb = np.linalg.norm(b)
        Nv = Ns[-1]
        if nb > 1e-8:
            b /= nb
            ang = math.acos(max(-1, min(1, float(np.dot(T[i - 1], T[i])))))
            Nv = np.array(Matrix.Rotation(ang, 3, Vector(b)) @ Vector(Nv))
        Ns.append(Nv / np.linalg.norm(Nv))
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    circ = 2 * math.pi * max(R.mean(), 1e-5)
    verts, uvs = [], []
    for i in range(n):
        Bv = np.cross(T[i], Ns[i])
        for k in range(sides + 1):
            a = 2 * math.pi * k / sides
            d = Ns[i] * math.cos(a) + Bv * math.sin(a)
            r = R[i] * (bump(P[i], d, i, k % sides) if bump else 1.0)
            verts.append(P[i] + d * r)
            uvs.append((k / sides, seg[i] / circ * uv_scale))
    faces = grid_faces(sides, n - 1)
    if cap:
        c = len(verts)
        verts.append(P[-1] + T[-1] * R[-1] * 0.3)
        uvs.append((0.5, seg[-1] / circ * uv_scale + 0.05))
        row = (n - 1) * (sides + 1)
        for k in range(sides):
            faces.append([row + k, row + k + 1, c])
    return np.array(verts), faces, uvs


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(np.asarray, (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def catmull(points, per=6):
    """Smooth curve through points (Catmull-Rom)."""
    P = [np.asarray(p, dtype=float) for p in points]
    P = [P[0] * 2 - P[1]] + P + [P[-1] * 2 - P[-2]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def frame(origin, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0):
    """Matrix placing a part built along +Y (normal +Z): roll about Y, then
    pitch (tip up toward +Z), then yaw about Z, then move to origin."""
    m = (Matrix.Translation(Vector(origin)) @ Matrix.Rotation(yaw, 4, "Z")
         @ Matrix.Rotation(pitch, 4, "X") @ Matrix.Rotation(roll, 4, "Y")
         @ Matrix.Scale(scale, 4))
    return m


def look_frame(origin, direction, roll=0.0, scale=1.0):
    """Matrix taking +Y to `direction` (normal roughly up)."""
    d = Vector(direction).normalized()
    yaw = math.atan2(-d.x, d.y)
    pitch = math.asin(max(-1, min(1, d.z)))
    return frame(origin, yaw, pitch, roll, scale)


# ---------------------------------------------------------------- noise

def _hash3(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 144665) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFF) / 65535.0


def vnoise3(p, seed=0):
    """Smooth value noise in [-1, 1] for points p (N,3)."""
    p = np.asarray(p, dtype=float)
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3 - 2 * f)
    out = np.zeros(len(p))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out += w * _hash3(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz, seed)
    return out * 2 - 1


def fbm3(p, octaves=4, seed=0, lac=2.03, gain=0.5):
    p = np.asarray(p, dtype=float)
    out, amp, tot = np.zeros(len(p)), 1.0, 0.0
    for o in range(octaves):
        out += amp * vnoise3(p, seed + o * 31)
        tot += amp
        p = p * lac
        amp *= gain
    return out / tot


# ---------------------------------------------------------------- export

def export_glb(path, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objects:
        o.select_set(True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(path), export_format="GLB", use_selection=True,
        export_apply=True, export_yup=True, export_texcoords=True,
        export_normals=True, export_materials="EXPORT",
        export_vertex_color="ACTIVE", export_all_vertex_colors=False,
        export_animations=False, export_extras=True,
    )
    print(f"  wrote {path} ({path.stat().st_size // 1024} KB)")
