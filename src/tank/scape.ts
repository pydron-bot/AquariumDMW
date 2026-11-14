import { BufferAttribute, Group, Matrix4, Mesh, MeshPhysicalMaterial, Object3D, TorusGeometry, Vector3 } from 'three';
import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { TANK, WATER_Y } from './dims';
import { materialFor } from './materials';
import { patch } from './shaders';
import { surfaceY } from './substrate';

const loader = new GLTFLoader();

export function loadGLB(name: string): Promise<GLTF> {
  return loader.loadAsync(`${import.meta.env.BASE_URL}assets/${name}`);
}

/** Swap the exporter's placeholder materials for ours, by name. */
export function dress(root: Object3D, shadows = { cast: true, receive: true }): void {
  root.traverse((o) => {
    const m = o as Mesh;
    if (!m.isMesh) return;
    const name = (Array.isArray(m.material) ? m.material[0] : m.material).name;
    if (!name) console.warn('unnamed material on', m.name, m.parent?.name);
    m.material = materialFor(name);
    m.castShadow = shadows.cast && !name.startsWith('fish_fin') && name !== 'led';
    m.receiveShadow = shadows.receive;
  });
}

/**
 * Where everything grows, read off the front photo (x: -0.25 left .. 0.25
 * right; z: -0.1275 back .. 0.1275 front). y = undefined sits on the sand;
 * a number is an epiphyte tied to the wood at that height.
 */
type Spot = [template: string, x: number, z: number, yawDeg: number, scale: number, y?: number];
export const LAYOUT: Spot[] = [
  // left: tall java fern in the back corner, grass in front, anubias
  ['javafern_tall_0', -0.215, -0.085, 20, 1.0],
  ['javafern_0', -0.2, -0.02, 70, 0.95],
  ['grass_0', -0.218, 0.075, 0, 1.0],
  ['grass_1', -0.19, 0.1, 40, 0.9],
  ['grass_2', -0.145, 0.105, 90, 0.75],
  ['grass_0', -0.06, 0.108, 160, 0.7],
  ['grass_1', 0.01, 0.11, 30, 0.55],
  ['anubias_0', -0.145, 0.045, 10, 1.15],
  ['anubias_small_0', -0.105, 0.08, 120, 0.9],
  ['crypt_wendtii_2', -0.2, 0.035, 200, 0.75],
  // centre: on and around the wood
  ['anubias_1', -0.03, 0.035, 200, 1.2, 0.072],
  ['anubias_2', 0.012, 0.012, 60, 1.0, 0.098],
  ['anubias_small_1', 0.05, 0.05, 300, 1.0],
  ['javafern_trident_0', -0.066, -0.02, 0, 1.0, 0.178],
  ['javafern_trident_1', -0.015, -0.035, 140, 0.85, 0.13],
  ['javafern_1', -0.1, -0.005, 250, 0.7, 0.118],
  ['najas_0', -0.005, -0.095, 0, 1.15],
  ['najas_1', 0.055, -0.1, 90, 1.05],
  ['najas_0', -0.08, -0.105, 200, 1.05],
  ['ludwigia_0', -0.035, -0.1, 0, 1.3],
  ['ludwigia_1', 0.025, -0.098, 120, 1.3],
  // right: a wall of moneywort, crypts in front
  ['bacopa_0', 0.165, -0.025, 0, 1.0],
  ['bacopa_1', 0.205, 0.03, 90, 0.95],
  ['bacopa_2', 0.135, -0.07, 200, 1.05],
  ['bacopa_0', 0.215, -0.08, 300, 1.1],
  ['bacopa_2', 0.1, -0.02, 30, 0.8],
  ['crypt_wendtii_0', 0.15, 0.085, 0, 1.0],
  ['crypt_wendtii_1', 0.2, 0.095, 120, 0.95],
  ['crypt_wendtii_2', 0.11, 0.06, 240, 0.85],
  ['crypt_flamingo_0', 0.075, 0.08, 40, 1.0],
  ['crypt_flamingo_1', 0.225, 0.065, 180, 0.9],
  ['crypt_flamingo_0', 0.125, 0.105, 260, 0.8],
];

/** Glass and surface limits for plants: leaves that would pass through a
 * pane are pressed flat against it, and stems that would break the
 * surface bend under it, as they do in the real tank. Each placement gets
 * its own copy of the geometry (the clamp depends on where it stands). */
const INSET = 0.004;
function keepInside(plant: Object3D): void {
  plant.updateMatrixWorld(true);
  const v = new Vector3();
  const inv = new Matrix4();
  plant.traverse((o) => {
    const m = o as Mesh;
    if (!m.isMesh) return;
    m.geometry = m.geometry.clone();
    const pos = m.geometry.attributes.position as BufferAttribute;
    inv.copy(m.matrixWorld).invert();
    let changed = false;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      const x = Math.max(-TANK.hx + INSET, Math.min(TANK.hx - INSET, v.x));
      const z = Math.max(-TANK.hz + INSET, Math.min(TANK.hz - INSET, v.z));
      const y = Math.min(WATER_Y - 0.003, v.y);
      if (x !== v.x || y !== v.y || z !== v.z) {
        v.set(x, y, z).applyMatrix4(inv);
        pos.setXYZ(i, v.x, v.y, v.z);
        changed = true;
      }
    }
    if (changed) m.geometry.computeVertexNormals();
  });
}

export interface Scape {
  root: Group;
  /** Meshes that block light (for the light-shaft depth map). */
  occluders: Object3D[];
  lightBar: Object3D | null;
}

export async function buildScape(): Promise<Scape> {
  const [plants, scape] = await Promise.all([loadGLB('plants.glb'), loadGLB('scape.glb')]);
  const root = new Group();
  root.name = 'scape';
  const occluders: Object3D[] = [];

  const templates = new Map<string, Object3D>();
  for (const child of plants.scene.children) templates.set(child.name, child);
  dress(plants.scene);

  for (const [name, x, z, yaw, scale, y] of LAYOUT) {
    const t = templates.get(name);
    if (!t) {
      console.warn('missing plant', name);
      continue;
    }
    const p = t.clone();
    p.position.set(x, y ?? surfaceY(x, z) - 0.004, z);
    p.rotation.set(0, (yaw * Math.PI) / 180, 0);
    p.scale.setScalar(scale);
    keepInside(p);
    root.add(p);
    occluders.push(p);
  }

  // Hardscape and hardware are authored in tank space.
  dress(scape.scene);
  let lightBar: Object3D | null = null;
  for (const child of [...scape.scene.children]) {
    root.add(child);
    if (child.name === 'light') lightBar = child;
    else if (child.name !== 'trim') occluders.push(child);
  }

  // Floating feeding ring, back right (see the top-down photo), with a few
  // stem cuttings drifting inside it.
  const ringMat = patch(
    new MeshPhysicalMaterial({ color: 0xcfd6d6, roughness: 0.2, transparent: true, opacity: 0.55, clearcoat: 1 }),
    { caustics: false },
  );
  const ring = new Mesh(new TorusGeometry(0.034, 0.0035, 10, 48), ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.position.set(0.15, WATER_Y + 0.001, -0.055);
  ring.name = 'feeding_ring';
  root.add(ring);
  const cutting = templates.get('ludwigia_1');
  if (cutting) {
    for (let k = 0; k < 3; k++) {
      const c = cutting.clone();
      c.scale.setScalar(0.35);
      c.rotation.set(Math.PI / 2 - 0.2, k * 2.1, 0.3 * k);
      c.position.set(0.15 + (k - 1) * 0.012, WATER_Y - 0.004, -0.055 + (k - 1) * 0.01);
      root.add(c);
    }
  }
  return { root, occluders, lightBar };
}

export const salviniaTemplate = async (): Promise<Mesh | null> => {
  const plants = await loadGLB('plants.glb');
  let found: Mesh | null = null;
  plants.scene.traverse((o) => {
    if (!found && (o as Mesh).isMesh && o.name.startsWith('salvinia')) found = o as Mesh;
  });
  return found;
};
