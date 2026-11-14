# Tank

A three.js tech demo: a detailed recreation of Derek's planted 10 gallon
(Imagitarium, 50 x 25.5 x 31 cm inside) with 12 White Cloud Mountain
Minnows. Built to see how far three.js can be pushed.

Dev server: `systemctl --user status tank-dev`, at http://hive1:5175 (tailnet).

## Commands

```sh
npm run assets              # Blender: rebuild every model and texture (~25 s)
blender -b --factory-startup -P blender/build.py -- --out public/assets --only plants,textures
npm run dev                 # vite on :5175
npm run typecheck
node tools/shot.mjs --gpu --views front,corner,left,top,macro,inside,fish --out shots
```

`tools/shot.mjs` renders headless (SwiftShader, ~10 fps at 1600x900) and
writes PNGs; `--eval "<js>"` runs against `window.__tank` first (e.g.
`window.__tank.life.feed(0, 0)`). URL params: `?q=high|medium|low`,
`view=front|corner|left|top|macro|inside|fish`, `ui=0`, `stats=1`,
`post=0` (no post-processing), `off=shafts,bloom,final`.

## Layout

| Path | What |
| --- | --- |
| `blender/build.py` | Entry point for the Blender pipeline |
| `blender/lib/plants.py` | Leaf/stem generators: Anubias, Java fern (+ trident), Crypt wendtii, Crypt 'Flamingo', dwarf sag, moneywort, Ludwigia, guppy grass, pennywort vines, Salvinia node |
| `blender/lib/hardscape.py` | Driftwood skeleton (laid out from the photos), dark rock, pebbles, pennywort paths |
| `blender/lib/fauna.py` | White Cloud (body, fins, eyes + painted side-view texture), ramshorn snail |
| `blender/lib/equipment.py` | Tank trim, LED bar + emissive LED map, sponge filter, internal filters, cables |
| `blender/lib/textures.py` | numpy-painted textures: sand, soil, bark, rock, pebble, leaves, roots, drops |
| `src/tank/shaders.ts` | The shared injection: caustics (inside the shadowed light loop), water absorption/scatter along the view ray, sway, leaf translucency |
| `src/post.ts` | Volumetric light shafts (ray-marched against a light-space depth map), DOF, bloom, grade |
| `src/tank/scape.ts` | Plant layout table (positions read off the front photo) |
| `src/tank/life.ts` | Boids school + swim shader, feeding, snails, bubbles, suspended particles |
| `src/tank/glass.ts` | Glass, edges, condensation (open top, no lid), water surface + total-internal-reflection mirror |
| `src/tank/salvinia.ts` | Floating mat (instanced) and root curtain (alpha cards) |

Units are metres. Tank space: origin at the centre of the tank floor,
+Z toward the front glass. The sand height formula lives in both
`hardscape.py` and `dims.ts`; keep them in step.

Textures are procedural for now. `docs/TEXTURES.md` lists the reference
photos that would replace them.
