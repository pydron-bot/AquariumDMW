import {
  ACESFilmicToneMapping,
  AgXToneMapping,
  Color,
  Object3D,
  PCFShadowMap,
  PerspectiveCamera,
  Plane,
  Raycaster,
  Scene,
  Vector2,
  Vector3,
  Timer,
  WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildPost, OCCLUDER_LAYER, type Post } from './post';
import { TANK, WATER_Y } from './tank/dims';
import { buildGlass, type Glass } from './tank/glass';
import { Life } from './tank/life';
import { initTextures, texturesLoaded } from './tank/materials';
import { buildRoom, type LightMode, type Room } from './tank/room';
import { buildSalvinia, findSalviniaGeometry } from './tank/salvinia';
import { buildScape, loadGLB } from './tank/scape';
import { U } from './tank/shaders';
import { buildSubstrate } from './tank/substrate';
import './style.css';

type Quality = 'high' | 'medium' | 'low';
type View = 'front' | 'corner' | 'left' | 'top' | 'macro' | 'fish' | 'inside';

const params = new URLSearchParams(location.search);
const coarse = matchMedia('(pointer: coarse)').matches;
let quality: Quality = (params.get('q') as Quality) || (coarse ? 'medium' : 'high');

const canvas = document.getElementById('c') as HTMLCanvasElement;
const renderer = new WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = PCFShadowMap;
renderer.toneMapping = params.get('tm') === 'aces' ? ACESFilmicToneMapping : AgXToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.info.autoReset = false;
initTextures(renderer);

const scene = new Scene();
scene.background = new Color(0x0d0b12);
const camera = new PerspectiveCamera(40, 1, 0.005, 12);
camera.layers.enable(OCCLUDER_LAYER);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.04;
controls.maxDistance = 2.2;
controls.target.set(0, 0.15, 0);

let room: Room;
let glass: Glass;
let life: Life;
let post: Post;
const clock = new Timer();

// ------------------------------------------------------------------ views

const VIEWS: Record<Exclude<View, 'fish'>, { pos: [number, number, number]; target: [number, number, number]; fov: number }> = {
  front: { pos: [0, 0.2, 0.6], target: [0, 0.165, 0], fov: 40 },
  corner: { pos: [0.4, 0.1, 0.42], target: [0.04, 0.15, 0], fov: 40 },
  left: { pos: [-0.46, 0.2, 0.3], target: [0.0, 0.15, 0], fov: 40 },
  top: { pos: [0, 0.72, 0.36], target: [0, 0.2, -0.02], fov: 40 },
  macro: { pos: [0.075, 0.13, 0.19], target: [0.03, 0.12, 0.0], fov: 32 },
  inside: { pos: [0.18, 0.11, 0.09], target: [-0.1, 0.15, -0.05], fov: 55 },
};

let view: View = (params.get('view') as View) || 'front';
let fly: { from: Vector3; to: Vector3; tFrom: Vector3; tTo: Vector3; fovFrom: number; fovTo: number; k: number } | null = null;

/** Pull the camera back on narrow (portrait) screens so the tank still fits. */
function fitScale(): number {
  const aspect = camera.aspect;
  return aspect < 1.5 ? Math.min(2.4, 1.5 / aspect) : 1;
}

function setView(v: View, instant = false): void {
  view = v;
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  controls.enabled = v !== 'fish';
  if (v === 'fish') return;
  const def = VIEWS[v];
  const t = new Vector3(...def.target);
  const p = new Vector3(...def.pos).sub(t).multiplyScalar(v === 'macro' || v === 'inside' ? 1 : fitScale()).add(t);
  if (instant) {
    camera.position.copy(p);
    controls.target.copy(t);
    camera.fov = def.fov;
    camera.updateProjectionMatrix();
    return;
  }
  fly = { from: camera.position.clone(), to: p, tFrom: controls.target.clone(), tTo: t, fovFrom: camera.fov, fovTo: def.fov, k: 0 };
}

// ------------------------------------------------------------------ quality

function applyQuality(q: Quality): void {
  quality = q;
  document.querySelectorAll<HTMLButtonElement>('[data-q]').forEach((b) => b.classList.toggle('on', b.dataset.q === q));
  const dpr = q === 'high' ? Math.min(2, devicePixelRatio) : q === 'medium' ? Math.min(1.25, devicePixelRatio) : 1;
  renderer.setPixelRatio(dpr);
  const scale = Number(params.get('renderScale') || 1);
  if (scale !== 1) renderer.setPixelRatio(dpr * scale);
  if (post) {
    const rt = post.composer.renderTarget1;
    rt.samples = rt.samples; // samples are fixed at creation; see rebuildPost
    post.shafts.steps = q === 'high' ? 40 : 22;
    post.shafts.enabledShafts = q !== 'low' && toggles.shafts;
    post.final.mat.uniforms.uDof.value = q !== 'low' && toggles.dof ? 1 : 0;
  }
  room?.sun.shadow.mapSize.setScalar(q === 'low' ? 1024 : 2048);
  if (room?.sun.shadow.map) {
    room.sun.shadow.map.dispose();
    room.sun.shadow.map = null as never;
  }
  glass?.setMirror(q !== 'low' && toggles.mirror);
  resize();
}

const toggles = { shafts: true, dof: true, bloom: true, caustics: true, mirror: true, water: true, stats: params.get('stats') === '1' };

function resize(): void {
  const w = innerWidth, h = innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  post?.setSize(w, h, renderer.getPixelRatio());
}

// ------------------------------------------------------------------ feeding & focus

const ray = new Raycaster();
const ndc = new Vector2();
const surfacePlane = new Plane(new Vector3(0, 1, 0), -WATER_Y);
let down: { x: number; y: number; t: number } | null = null;
canvas.addEventListener('pointerdown', (e) => (down = { x: e.clientX, y: e.clientY, t: performance.now() }));
canvas.addEventListener('pointerup', (e) => {
  if (!down) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  const quick = performance.now() - down.t < 350;
  down = null;
  if (moved > 6 || !quick) return;
  ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.ray.intersectPlane(surfacePlane, new Vector3());
  if (hit && Math.abs(hit.x) < TANK.hx && Math.abs(hit.z) < TANK.hz) life?.feed(hit.x, hit.z);
});
function feedRandom(): void {
  life?.feed((Math.random() - 0.5) * 0.3, (Math.random() - 0.5) * 0.12);
}

let focusTargets: Object3D[] = [];
let focus = 0.6;
let frameNo = 0;
function updateFocus(): void {
  let want = focus;
  if (view === 'fish') {
    want = camera.position.distanceTo(life.fishPosition(life.focusFish, new Vector3()));
  } else if (frameNo % 6 === 0) {
    ray.setFromCamera(new Vector2(0, 0), camera);
    const hits = ray.intersectObjects(focusTargets, true);
    want = hits.length ? hits[0].distance : camera.position.distanceTo(controls.target);
  } else return;
  focus += (want - focus) * (view === 'fish' ? 0.3 : 0.25);
  post.final.mat.uniforms.uFocus.value = focus;
}

// ------------------------------------------------------------------ fish cam

const fishCamPos = new Vector3(), fishCamLook = new Vector3();
function updateFishCam(dt: number): void {
  const p = life.fishPosition(life.focusFish, new Vector3());
  const v = life.fishVelocity(life.focusFish, new Vector3());
  const back = v.lengthSq() > 1e-8 ? v.clone().normalize() : new Vector3(1, 0, 0);
  const side = new Vector3().crossVectors(back, new Vector3(0, 1, 0)).normalize();
  const want = p.clone().addScaledVector(back, -0.06).addScaledVector(side, 0.035).add(new Vector3(0, 0.012, 0));
  // stay inside the water
  want.x = Math.max(-TANK.hx + 0.01, Math.min(TANK.hx - 0.01, want.x));
  want.y = Math.max(0.05, Math.min(WATER_Y - 0.01, want.y));
  want.z = Math.max(-TANK.hz + 0.01, Math.min(TANK.hz - 0.01, want.z));
  const k = 1 - Math.exp(-dt * 2.5);
  fishCamPos.lerp(want, k);
  fishCamLook.lerp(p, 1 - Math.exp(-dt * 6));
  camera.position.copy(fishCamPos);
  camera.lookAt(fishCamLook);
  if (camera.fov !== 50) {
    camera.fov = 50;
    camera.updateProjectionMatrix();
  }
}

// ------------------------------------------------------------------ stats

const statsEl = document.getElementById('stats')!;
let fpsAcc = 0, fpsN = 0, fpsShown = 0, lastStat = 0;
function updateStats(dt: number, now: number): void {
  fpsAcc += dt;
  fpsN++;
  if (now - lastStat > 500) {
    fpsShown = fpsN / fpsAcc;
    fpsAcc = 0;
    fpsN = 0;
    lastStat = now;
    if (toggles.stats) {
      const i = renderer.info;
      statsEl.textContent =
        `${fpsShown.toFixed(0)} fps · ${(1000 / fpsShown).toFixed(1)} ms\n` +
        `${i.render.calls} draws · ${(i.render.triangles / 1000).toFixed(0)}k tris\n` +
        `${i.programs?.length ?? 0} programs · ${i.memory.textures} textures\n` +
        `${innerWidth}×${innerHeight} @${renderer.getPixelRatio().toFixed(2)}x · ${quality}`;
    }
  }
}

// ------------------------------------------------------------------ loop

function frame(): void {
  clock.update();
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.getElapsed();
  U.uTime.value = t;
  frameNo++;
  life.update(dt, t);
  if (fly) {
    fly.k = Math.min(1, fly.k + dt / 1.3);
    const e = fly.k < 0.5 ? 4 * fly.k ** 3 : 1 - (-2 * fly.k + 2) ** 3 / 2;
    camera.position.lerpVectors(fly.from, fly.to, e);
    controls.target.lerpVectors(fly.tFrom, fly.tTo, e);
    camera.fov = fly.fovFrom + (fly.fovTo - fly.fovFrom) * e;
    camera.updateProjectionMatrix();
    if (fly.k >= 1) fly = null;
  }
  if (view === 'fish') updateFishCam(dt);
  else controls.update(dt);
  updateFocus();
  renderer.info.reset();
  if (params.get('post') === '0') renderer.render(scene, camera);
  else post.composer.render(dt);
  updateStats(dt, performance.now());
}

// ------------------------------------------------------------------ boot

async function boot(): Promise<void> {
  const status = document.getElementById('loading')!;
  room = buildRoom(renderer, scene);
  scene.add(room.root);
  status.textContent = 'Loading plants and wood…';
  const [scape, fauna, plantsGLB] = await Promise.all([buildScape(), loadGLB('fauna.glb'), loadGLB('plants.glb')]);
  scene.add(scape.root);
  for (const o of scape.occluders) o.traverse((c) => c.layers.enable(OCCLUDER_LAYER));
  const substrate = buildSubstrate();
  scene.add(substrate);
  const salGeo = findSalviniaGeometry(plantsGLB.scene);
  if (salGeo) scene.add(buildSalvinia(salGeo));
  life = new Life(fauna.scene);
  scene.add(life.root);
  glass = buildGlass(true);
  scene.add(glass.root);
  focusTargets = [scape.root, substrate, life.root];

  post = buildPost(renderer, scene, camera, quality === 'low' ? 0 : 4);
  for (const off of (params.get('off') ?? '').split(',')) {
    if (off === 'shafts') post.shafts.enabledShafts = false;
    if (off === 'bloom') post.bloom.enabled = false;
    if (off === 'final') post.final.enabled = false;
  }
  setView(view, true);
  applyQuality(quality);
  if ((params.get('off') ?? '').includes('shafts')) post.shafts.enabledShafts = false;
  fishCamPos.copy(life.fishPosition(0, new Vector3()));
  status.textContent = 'Loading textures…';
  await texturesLoaded();
  // Compile everything before the first frame so nothing hitches later.
  await renderer.compileAsync(scene, camera);
  status.remove();
  document.body.classList.add('ready');
  renderer.setAnimationLoop(frame);
  (window as unknown as { __tank: unknown }).__tank = {
    ready: true, setView, applyQuality, life, renderer, scene, camera, U, toggles, room, post,
    /** Render a frame and read it back in the same task (headless screenshots). */
    capture: () => {
      frame();
      return canvas.toDataURL('image/png');
    },
  };
}

// ------------------------------------------------------------------ UI

function bindUI(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) =>
    b.addEventListener('click', () => {
      if (b.dataset.view === 'fish' && view === 'fish') life.focusFish = (life.focusFish + 1) % 12;
      setView(b.dataset.view as View);
    }),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-q]').forEach((b) =>
    b.addEventListener('click', () => applyQuality(b.dataset.q as Quality)),
  );
  document.querySelectorAll<HTMLButtonElement>('[data-light]').forEach((b) =>
    b.addEventListener('click', () => {
      room.setMode(b.dataset.light as LightMode);
      document.querySelectorAll<HTMLButtonElement>('[data-light]').forEach((x) => x.classList.toggle('on', x === b));
    }),
  );
  document.getElementById('feed')!.addEventListener('click', feedRandom);
  document.querySelectorAll<HTMLInputElement>('[data-fx]').forEach((el) => {
    const key = el.dataset.fx as keyof typeof toggles;
    el.checked = toggles[key];
    el.addEventListener('change', () => {
      toggles[key] = el.checked;
      if (key === 'bloom') post.bloom.enabled = el.checked;
      if (key === 'caustics') U.uCaustic.value = el.checked ? 1 : 0;
      if (key === 'water') U.uWaterOn.value = el.checked ? 1 : 0;
      if (key === 'stats') statsEl.hidden = !el.checked;
      applyQuality(quality);
    });
  });
  statsEl.hidden = !toggles.stats;
  document.getElementById('menu-toggle')!.addEventListener('click', () => document.body.classList.toggle('menu-open'));
  if (params.get('ui') === '0') document.body.classList.add('no-ui');
}

addEventListener('resize', resize);
resize();
bindUI();
boot().catch((e) => {
  console.error(e);
  document.getElementById('loading')!.textContent = 'Failed to load: ' + (e as Error).message;
});
