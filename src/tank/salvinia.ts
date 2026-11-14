import {
  DoubleSide,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshDepthMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  Quaternion,
  RGBADepthPacking,
  Vector3,
  type BufferGeometry,
} from 'three';
import { OCCLUDER_LAYER } from '../post';
import { TANK, WATER_Y, rng } from './dims';
import { materialFor, tex } from './materials';
import { patch } from './shaders';

/**
 * The floating Salvinia mat (top-down photo: nearly complete cover with a
 * few gaps and the feeding ring) and its curtain of hairy, golden-brown
 * submerged fronds hanging 2-8 cm.
 */
export function buildSalvinia(nodeGeometry: BufferGeometry): Group {
  const g = new Group();
  g.name = 'salvinia';
  const r = rng(77);
  const { hx, hz } = TANK;

  // Gaps: feeding ring, the sponge filter's outlet, a few open patches.
  const holes: [number, number, number][] = [
    [0.15, -0.055, 0.037],
    [-0.15, -0.065, 0.028],
    [0.03, 0.02, 0.012],
    [-0.06, 0.07, 0.01],
    [0.2, 0.09, 0.012],
  ];
  const inHole = (x: number, z: number) => holes.some(([hxp, hzp, rr]) => (x - hxp) ** 2 + (z - hzp) ** 2 < rr * rr);

  const pts: [number, number][] = [];
  const step = 0.0078;
  for (let x = -hx + 0.004; x < hx - 0.004; x += step) {
    for (let z = -hz + 0.004; z < hz - 0.004; z += step) {
      const px = x + (r() - 0.5) * step * 1.1;
      const pz = z + (r() - 0.5) * step * 1.1;
      if (inHole(px, pz) || r() < 0.06) continue;
      pts.push([px, pz]);
    }
  }
  const mat = materialFor('leaf_salvinia');
  const mesh = new InstancedMesh(nodeGeometry, mat, pts.length);
  const m = new Matrix4(), q = new Quaternion(), s = new Vector3(), p = new Vector3(), up = new Vector3(0, 1, 0);
  pts.forEach(([x, z], i) => {
    q.setFromAxisAngle(up, r() * Math.PI * 2);
    s.setScalar(0.75 + r() * 0.55);
    // Sitting in the surface: the undersides are wet, so they show through
    // the mirror of the surface seen from below.
    p.set(x, WATER_Y - 0.0018, z);
    m.compose(p, q, s);
    mesh.setMatrixAt(i, m);
  });
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = 'salvinia_mat';
  mesh.layers.enable(OCCLUDER_LAYER);
  g.add(mesh);

  // Root curtain: crossed alpha cards under the nodes.
  const card = new PlaneGeometry(0.024, 1, 1, 8);
  card.translate(0, -0.5, 0);
  const uv = card.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setY(i, 1 - uv.getY(i));
  const crossed = mergeCrossed(card);
  const rootsTex = tex('salvinia_roots.png');
  const rootsMat = patch(
    new MeshStandardMaterial({
      map: rootsTex,
      alphaTest: 0.36,
      alphaToCoverage: true,
      color: 0xf0d9a0,
      side: DoubleSide,
      roughness: 0.8,
    }),
    { caustics: true, sway: 0.006, swayMode: 'hang', translucent: true },
  );
  const n = Math.floor(pts.length * 0.42);
  const roots = new InstancedMesh(crossed, rootsMat, n);
  for (let i = 0; i < n; i++) {
    const [x, z] = pts[Math.floor(r() * pts.length)];
    // Longer near the middle of the mat, shorter at the glass.
    const edge = Math.min(hx - Math.abs(x), hz - Math.abs(z));
    const len = (0.012 + r() ** 1.7 * 0.07) * Math.min(1, 0.5 + edge * 14);
    q.setFromAxisAngle(up, r() * Math.PI);
    s.set(0.7 + r() * 0.6, len, 1);
    p.set(x, WATER_Y - 0.0005, z);
    m.compose(p, q, s);
    roots.setMatrixAt(i, m);
  }
  roots.castShadow = true;
  roots.customDepthMaterial = new MeshDepthMaterial({ depthPacking: RGBADepthPacking, map: rootsTex, alphaTest: 0.3 });
  roots.name = 'salvinia_roots';
  roots.renderOrder = 2;
  g.add(roots);
  return g;
}

function mergeCrossed(card: PlaneGeometry): BufferGeometry {
  const a = card.clone();
  const b = card.clone().rotateY(Math.PI / 2);
  const pos = new Float32Array(a.attributes.position.count * 6);
  const nor = new Float32Array(pos.length);
  const uvs = new Float32Array(a.attributes.uv.count * 4);
  pos.set(a.attributes.position.array as Float32Array, 0);
  pos.set(b.attributes.position.array as Float32Array, a.attributes.position.array.length);
  nor.set(a.attributes.normal.array as Float32Array, 0);
  nor.set(b.attributes.normal.array as Float32Array, a.attributes.normal.array.length);
  uvs.set(a.attributes.uv.array as Float32Array, 0);
  uvs.set(b.attributes.uv.array as Float32Array, a.attributes.uv.array.length);
  const ia = Array.from(a.index!.array);
  const off = a.attributes.position.count;
  const idx = [...ia, ...Array.from(b.index!.array).map((i) => i + off)];
  const g = a.clone();
  g.setAttribute('position', new (a.attributes.position.constructor as any)(pos, 3));
  g.setAttribute('normal', new (a.attributes.normal.constructor as any)(nor, 3));
  g.setAttribute('uv', new (a.attributes.uv.constructor as any)(uvs, 2));
  g.setIndex(idx);
  return g;
}

export function findSalviniaGeometry(root: { traverse(cb: (o: unknown) => void): void }): BufferGeometry | null {
  let geo: BufferGeometry | null = null;
  root.traverse((o) => {
    const mesh = o as Mesh;
    if (!geo && mesh.isMesh && mesh.name.startsWith('salvinia')) geo = mesh.geometry;
  });
  return geo;
}
