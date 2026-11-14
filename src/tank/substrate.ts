import {
  BufferAttribute,
  BufferGeometry,
  Group,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  type Texture,
} from 'three';
import { TANK, sandY } from './dims';
import { tex } from './materials';
import { patch } from './shaders';

/** Sand over aquasoil: a displaced heightfield plus the cut faces seen
 * through the glass (sand on top, dark soil underneath). The mix is a
 * per-vertex `mask` (0 sand .. 1 soil) the shader blends textures by. */

function hash(x: number, y: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
}
function vnoise(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy);
  const a = hash(ix, iy), b = hash(ix + 1, iy), c = hash(ix, iy + 1), d = hash(ix + 1, iy + 1);
  return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy;
}
function fbm(x: number, y: number): number {
  return vnoise(x, y) * 0.55 + vnoise(x * 2.1, y * 2.1) * 0.28 + vnoise(x * 4.3, y * 4.3) * 0.17;
}

/** Final surface height incl. small dunes and divots. */
export function surfaceY(x: number, z: number): number {
  return sandY(x, z) + (fbm(x * 18 + 3, z * 18) - 0.5) * 0.006 + (fbm(x * 70, z * 70) - 0.5) * 0.0012;
}

function soilMask(x: number, z: number): number {
  const back = (-z + TANK.hz) / (2 * TANK.hz);
  const n = fbm(x * 9 + 11, z * 9 - 4);
  // Soil shows at the back, under the wood and in patches.
  const wood = Math.exp(-((x + 0.05) ** 2 + (z + 0.0) ** 2) / 0.003);
  let m = (n - 0.48) * 4 + (back - 0.55) * 2.2 + wood * 1.2;
  return Math.min(1, Math.max(0, m));
}

function material(): MeshStandardMaterial {
  const sand = tex('sand_color.jpg', true, true);
  const soil = tex('soil_color.jpg', true, true);
  const m = new MeshStandardMaterial({
    map: sand,
    normalMap: tex('sand_normal.png', false, true),
    roughness: 0.95,
  });
  m.normalScale.setScalar(1.2);
  m.onBeforeCompile = (s) => {
    s.uniforms.tSoil = { value: soil as Texture };
    s.vertexShader = s.vertexShader
      .replace('#include <common>', '#include <common>\nattribute float mask;\nvarying float vMask;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvMask = mask;');
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D tSoil;\nvarying float vMask;')
      .replace(
        '#include <map_fragment>',
        `#include <map_fragment>
        vec4 soilC = texture2D(tSoil, vMapUv * 1.6);
        diffuseColor.rgb = mix(diffuseColor.rgb, soilC.rgb, smoothstep(0.15, 0.85, vMask));`,
      );
  };
  return patch(m, { caustics: true });
}

export function buildSubstrate(): Group {
  const g = new Group();
  g.name = 'substrate';
  const W = TANK.hx * 2, D = TANK.hz * 2;
  const TILE = 0.075; // sand texture covers 7.5 cm
  const geo = new PlaneGeometry(W, D, 220, 112);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position as BufferAttribute;
  const uv = geo.attributes.uv as BufferAttribute;
  const mask = new Float32Array(pos.count);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    pos.setY(i, surfaceY(x, z));
    uv.setXY(i, x / TILE, z / TILE);
    mask[i] = soilMask(x, z);
  }
  geo.setAttribute('mask', new BufferAttribute(mask, 1));
  geo.computeVertexNormals();
  const mat = material();
  const top = new Mesh(geo, mat);
  top.receiveShadow = true;
  top.name = 'sand';
  g.add(top);

  // Cut faces against the glass: sand down to ~1.4 cm, soil below.
  const sides: [number, number, number, number][] = [
    [-TANK.hx, TANK.hz, TANK.hx, TANK.hz], // front
    [TANK.hx, -TANK.hz, -TANK.hx, -TANK.hz], // back
    [-TANK.hx, -TANK.hz, -TANK.hx, TANK.hz], // left
    [TANK.hx, TANK.hz, TANK.hx, -TANK.hz], // right
  ];
  for (const [x0, z0, x1, z1] of sides) {
    const n = 160, rows = 12;
    const verts: number[] = [], uvs: number[] = [], masks: number[] = [], idx: number[] = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      const x = x0 + (x1 - x0) * t, z = z0 + (z1 - z0) * t;
      const top = surfaceY(x, z);
      const along = Math.hypot(x - x0, z - z0);
      for (let j = 0; j <= rows; j++) {
        const y = (top * j) / rows;
        verts.push(x, y, z);
        uvs.push(along / TILE, y / TILE);
        const depth = top - y;
        const wav = 0.012 + 0.004 * Math.sin(along * 90) + 0.003 * fbm(along * 40, 1);
        masks.push(Math.max(soilMask(x, z) * (depth < 0.004 ? 1 : 0), depth > wav ? 1 : depth / wav * 0.2));
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < rows; j++) {
        const a = i * (rows + 1) + j, b = a + rows + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    const sg = new BufferGeometry();
    sg.setAttribute('position', new BufferAttribute(new Float32Array(verts), 3));
    sg.setAttribute('uv', new BufferAttribute(new Float32Array(uvs), 2));
    sg.setAttribute('mask', new BufferAttribute(new Float32Array(masks), 1));
    sg.setIndex(idx);
    sg.computeVertexNormals();
    const side = new Mesh(sg, mat);
    side.receiveShadow = true;
    g.add(side);
  }
  return g;
}
