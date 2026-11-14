import {
  Color,
  DoubleSide,
  LinearSRGBColorSpace,
  Material,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  RepeatWrapping,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  type WebGLRenderer,
} from 'three';
import { patch } from './shaders';

const loader = new TextureLoader();
const cache = new Map<string, Texture>();
const pending: Promise<unknown>[] = [];

/** Resolves when every texture requested so far has loaded. */
export function texturesLoaded(): Promise<unknown> {
  return Promise.all(pending);
}
let maxAniso = 8;

export function initTextures(renderer: WebGLRenderer): void {
  maxAniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());
}

/** Loads public/assets/tex/<name> (glTF UV convention: flipY off). */
export function tex(name: string, color = true, repeat = false): Texture {
  const key = `${name}|${color}|${repeat}`;
  let t = cache.get(key);
  if (!t) {
    let done: (v?: unknown) => void = () => {};
    pending.push(new Promise((r) => (done = r)));
    t = loader.load(`${import.meta.env.BASE_URL}assets/tex/${name}`, done, undefined, (e) => {
      console.error('texture failed', name, e);
      done();
    });
    t.flipY = false;
    t.colorSpace = color ? SRGBColorSpace : LinearSRGBColorSpace;
    t.anisotropy = maxAniso;
    if (repeat) t.wrapS = t.wrapT = RepeatWrapping;
    cache.set(key, t);
  }
  return t;
}

function leaf(name: string, roughness: number, opts: { sway?: number; normal?: number } = {}): Material {
  const m = new MeshStandardMaterial({
    map: tex(`leaf_${name}.jpg`),
    normalMap: tex(`leaf_${name}_n.png`, false),
    roughness,
    side: DoubleSide,
    vertexColors: true,
  });
  m.normalScale.setScalar(opts.normal ?? 0.8);
  return patch(m, { caustics: true, sway: opts.sway ?? 0.9, translucent: true });
}

function plain(name: string, roughness: number, sway = 0): Material {
  return patch(
    new MeshStandardMaterial({ map: tex(`${name}.jpg`, true, true), roughness, vertexColors: true }),
    { caustics: true, sway },
  );
}

/** The material for a GLB material name, built on first use. */
const table: Record<string, () => Material> = {
  leaf_anubias: () => leaf('anubias', 0.32, { sway: 0.25 }),
  leaf_javafern: () => leaf('javafern', 0.55, { sway: 0.9 }),
  leaf_crypt_wendtii: () => leaf('crypt_wendtii', 0.5, { sway: 0.6 }),
  leaf_crypt_flamingo: () => leaf('crypt_flamingo', 0.5, { sway: 0.6 }),
  leaf_bacopa: () => leaf('bacopa', 0.45, { sway: 0.35 }),
  leaf_ludwigia: () => leaf('ludwigia', 0.5, { sway: 0.35 }),
  leaf_grass: () => leaf('grass', 0.55, { sway: 2.2, normal: 0.3 }),
  leaf_najas: () => leaf('najas', 0.6, { sway: 0.45, normal: 0.2 }),
  leaf_pennywort: () => leaf('pennywort', 0.4, { sway: 0.08 }),
  leaf_salvinia: () => {
    const m = new MeshStandardMaterial({
      map: tex('leaf_salvinia.jpg'),
      normalMap: tex('leaf_salvinia_n.png', false),
      roughness: 0.85,
      side: DoubleSide,
      vertexColors: true,
    });
    return patch(m, { caustics: false, sway: 0.0012, swayMode: 'float', translucent: true });
  },
  stem_green: () => plain('stem_green', 0.6, 0.6),
  stem_red: () => plain('stem_red', 0.6, 0.6),
  rhizome: () => plain('rhizome', 0.7, 0.2),
  roots_white: () => plain('roots_white', 0.8, 0.6),
  bark: () => {
    const m = new MeshStandardMaterial({
      map: tex('bark_color.jpg', true, true),
      normalMap: tex('bark_normal.png', false, true),
      roughness: 0.82,
      vertexColors: true,
    });
    m.normalScale.setScalar(1.4);
    return patch(m, { caustics: true });
  },
  rock_dark: () => {
    const m = new MeshStandardMaterial({
      map: tex('rock_color.jpg', true, true),
      normalMap: tex('rock_normal.png', false, true),
      roughness: 0.9,
      vertexColors: true,
    });
    return patch(m, { caustics: true });
  },
  pebble: () => {
    const m = new MeshPhysicalMaterial({
      map: tex('pebble_color.jpg', true, true),
      normalMap: tex('pebble_normal.png', false, true),
      roughness: 0.45,
      clearcoat: 0.4,
      clearcoatRoughness: 0.3,
      vertexColors: true,
    });
    return patch(m, { caustics: true });
  },
  trim: () => patch(new MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.38 }), {}),
  light_body: () => patch(new MeshStandardMaterial({ color: 0x1a1b1e, roughness: 0.45, metalness: 0.4 }), {}),
  metal: () => patch(new MeshStandardMaterial({ color: 0x9aa0a6, roughness: 0.3, metalness: 0.9 }), {}),
  led: () => {
    const m = new MeshBasicMaterial({ map: tex('led.png'), toneMapped: false });
    m.color = new Color(9, 9, 9);
    return m;
  },
  green_plastic: () => patch(new MeshStandardMaterial({ color: 0x4fae6a, roughness: 0.35 }), { caustics: true }),
  green_tube: () =>
    patch(
      new MeshPhysicalMaterial({
        color: 0x63c27e,
        roughness: 0.15,
        transparent: true,
        opacity: 0.72,
        clearcoat: 1,
        side: DoubleSide,
        depthWrite: false,
      }),
      { caustics: true },
    ),
  sponge: () => {
    const m = new MeshStandardMaterial({
      map: tex('sponge_color.jpg', true, true),
      normalMap: tex('sponge_normal.png', false, true),
      roughness: 1,
    });
    m.map!.repeat.set(3, 2);
    return patch(m, { caustics: true });
  },
  black_plastic: () => patch(new MeshStandardMaterial({ color: 0x101112, roughness: 0.55 }), { caustics: true }),
  suction_cup: () =>
    patch(
      new MeshPhysicalMaterial({ color: 0xdfe6e6, roughness: 0.25, transparent: true, opacity: 0.55, depthWrite: false }),
      {},
    ),
  cable: () => patch(new MeshStandardMaterial({ color: 0x0c0c0d, roughness: 0.45 }), { caustics: true }),
  fish_body: () => {
    const m = new MeshPhysicalMaterial({
      map: tex('fish.png'),
      roughness: 0.32,
      clearcoat: 0.5,
      clearcoatRoughness: 0.25,
      iridescence: 0.55,
      iridescenceIOR: 1.6,
      iridescenceThicknessRange: [250, 520],
      side: DoubleSide,
    });
    return patch(m, { caustics: true });
  },
  fish_fin: () => {
    const m = new MeshStandardMaterial({
      map: tex('fish.png'),
      roughness: 0.5,
      transparent: true,
      side: DoubleSide,
      depthWrite: false,
    });
    return patch(m, { caustics: true, translucent: true });
  },
  fish_eye: () =>
    patch(
      new MeshPhysicalMaterial({ map: tex('fish_eye.jpg'), roughness: 0.08, clearcoat: 1, clearcoatRoughness: 0.02 }),
      { caustics: true },
    ),
  snail_shell: () =>
    patch(
      new MeshPhysicalMaterial({ map: tex('snail_shell.jpg', true, true), roughness: 0.35, clearcoat: 0.6 }),
      { caustics: true },
    ),
  snail_body: () => patch(new MeshStandardMaterial({ color: 0x5c4a3e, roughness: 0.4 }), { caustics: true }),
};

const built = new Map<string, Material>();

export function materialFor(name: string): Material {
  let m = built.get(name);
  if (!m) {
    const make = table[name];
    if (!make) {
      console.warn(`no material for "${name}"`, new Error().stack?.split('\n')[2]);
      m = patch(new MeshStandardMaterial({ color: 0xff00ff }), {});
    } else {
      m = make();
    }
    m.name = name;
    built.set(name, m);
  }
  return m;
}

/** A fresh, unshared instance (for meshes that patch it further, e.g. the fish). */
export function makeMaterial(name: string): Material {
  const m = table[name]?.() ?? patch(new MeshStandardMaterial({ color: 0xff00ff }), {});
  m.name = name;
  return m;
}

export function allMaterials(): Material[] {
  return [...built.values()];
}
