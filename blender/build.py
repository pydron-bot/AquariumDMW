"""Builds every asset: blender -b --factory-startup -P blender/build.py -- --out public/assets [--only part,...]

Parts: textures, plants, scape, fauna. Deterministic: same seeds, same files.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import equipment, fauna, hardscape, plants, textures  # noqa: E402
from lib.util import export_glb, reset  # noqa: E402


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out, only = Path("public/assets"), None
    for i, k in enumerate(a):
        if k == "--out":
            out = Path(a[i + 1])
        if k == "--only":
            only = set(a[i + 1].split(","))
    return out, only


def build_textures(out):
    t = out / "tex"
    textures.sand(t)
    textures.soil(t)
    textures.bark(t)
    textures.rock(t)
    textures.pebble(t)
    textures.condensation(t)
    textures.sponge(t)
    textures.desk(t)
    textures.salvinia_roots(t)
    textures.all_leaves(t)
    fauna.fish_texture(t)
    fauna.snail_texture(t)
    equipment.led_texture(t)


def build_plants(out):
    reset()
    objs = []
    specs = [
        ("anubias", lambda s: plants.anubias(s, 8, 1.0), 3),
        ("anubias_small", lambda s: plants.anubias(s, 5, 0.7), 2),
        ("javafern", lambda s: plants.javafern(s, 9, 1.0), 2),
        ("javafern_tall", lambda s: plants.javafern(s, 11, 1.45), 1),
        ("javafern_trident", lambda s: plants.javafern(s, 10, 0.9, trident=True), 2),
        ("crypt_wendtii", lambda s: plants.crypt(s, 11, 1.0), 3),
        ("crypt_flamingo", lambda s: plants.crypt(s, 8, 1.0, flamingo=True), 2),
        ("grass", lambda s: plants.grass(s, 14, 1.0), 3),
        ("bacopa", lambda s: plants.stem_bunch(s, "bacopa", 8, 0.2), 3),
        ("ludwigia", lambda s: plants.stem_bunch(s, "ludwigia", 6, 0.17), 2),
        ("najas", lambda s: plants.stem_bunch(s, "najas", 7, 0.22), 2),
    ]
    for name, fn, variants in specs:
        for k in range(variants):
            seed = 1000 + sum(map(ord, name)) * 7 + k * 13
            o = fn(seed).build(f"{name}_{k}")
            objs.append(o)
    objs.append(plants.salvinia_node(5).build("salvinia"))
    export_glb(out / "plants.glb", objs)


def build_scape(out):
    reset()
    objs = [
        hardscape.driftwood().build("driftwood"),
        hardscape.rocks().build("rocks"),
        hardscape.pennywort().build("pennywort"),
        equipment.tank_trim().build("trim", smooth=False),
        equipment.light_bar().build("light", smooth=False),
        equipment.sponge_filter().build("sponge_filter"),
        equipment.internal_filters().build("internal_filters"),
    ]
    export_glb(out / "scape.glb", objs)


def build_fauna(out):
    reset()
    body, fins, eye = fauna.fish()
    objs = [body.build("fish_body"), fins.build("fish_fins"), eye.build("fish_eyes"),
            fauna.ramshorn().build("snail")]
    export_glb(out / "fauna.glb", objs)


def main():
    out, only = args()
    steps = [("textures", build_textures), ("plants", build_plants), ("scape", build_scape), ("fauna", build_fauna)]
    for name, fn in steps:
        if only and name not in only:
            continue
        t0 = time.time()
        print(f"[{name}]")
        fn(out)
        print(f"[{name}] {time.time() - t0:.1f}s")


main()
