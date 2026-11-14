/** Tank geometry (metres, three.js space: +Y up, +Z toward the front glass,
 * origin at the centre of the tank floor inside the glass). An Imagitarium
 * 10 gallon: 50 x 25.5 x 31 cm inside. */
export const TANK = { hx: 0.25, hz: 0.1275, h: 0.31, glass: 0.004 };

/** Water surface: about 3.4 cm below the rim, like the photos. */
export const WATER_Y = 0.276;

/** Sand level at (x, z). Same formula as blender/lib/hardscape.py
 * substrate_height (Blender y = -z); keep them in step. */
export function sandY(x: number, z: number): number {
  const back = (-z + TANK.hz) / (2 * TANK.hz);
  return 0.028 + 0.024 * back + 0.008 * Math.exp(-((x + 0.06) ** 2) / 0.004);
}

/** Small deterministic PRNG so layouts don't change between loads. */
export function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1_000_000) / 1_000_000;
  };
}
