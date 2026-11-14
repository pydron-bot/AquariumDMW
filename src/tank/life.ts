import {
  AdditiveBlending,
  BufferAttribute,
  BufferGeometry,
  CircleGeometry,
  Color,
  DynamicDrawUsage,
  Group,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshStandardMaterial,
  Object3D,
  Points,
  PointsMaterial,
  Quaternion,
  SphereGeometry,
  Vector3,
  type Material,
} from 'three';
import { OCCLUDER_LAYER } from '../post';
import { TANK, WATER_Y, rng } from './dims';
import { makeMaterial, materialFor } from './materials';
import { patch } from './shaders';
import { surfaceY } from './substrate';

const FISH = 12;
const R = rng(1234);

/** Swim wave, bent in the fish's own space before the instance transform:
 * amplitude grows toward the tail; aSwim = (phase, amplitude, turn). */
function swimPatch(mat: Material): Material {
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (s, r) => {
    s.vertexShader = s.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec3 aSwim;')
      .replace(
        '#include <begin_vertex>',
        `#include <begin_vertex>
        {
          float sN = clamp((0.018 - transformed.z) / 0.036, 0.0, 1.0);
          float a = aSwim.y * (0.12 + sN * sN * 1.1);
          transformed.x += a * sin(sN * 5.2 - aSwim.x) + aSwim.z * sN * sN * 0.014;
        }`,
      );
    prev.call(mat, s, r);
  };
  const key = mat.customProgramCacheKey.bind(mat);
  mat.customProgramCacheKey = () => key() + '|swim';
  return mat;
}

/** Spheres the school steers around: the driftwood skeleton and the
 * densest plant masses (three.js space). */
const OBSTACLES: [number, number, number, number][] = [
  // driftwood (blender (x, y, z) -> three (x, z, -y))
  [-0.07, 0.03, -0.012, 0.045], [-0.062, 0.08, -0.004, 0.04], [-0.056, 0.128, -0.012, 0.03],
  [-0.064, 0.168, -0.022, 0.024], [-0.004, 0.115, -0.04, 0.025], [0.012, 0.178, -0.052, 0.02],
  [0.025, 0.108, -0.028, 0.026], [0.085, 0.158, -0.05, 0.024], [0.128, 0.205, -0.068, 0.02],
  [0.105, 0.198, -0.075, 0.016], [0.045, 0.066, 0.0, 0.024], [0.098, 0.082, -0.028, 0.02],
  // plant masses
  [0.17, 0.11, -0.01, 0.06], [0.2, 0.14, 0.03, 0.05], [-0.2, 0.08, -0.05, 0.05],
  [-0.14, 0.05, 0.05, 0.05], [-0.168, 0.12, -0.05, 0.02],
  // the pump on the left glass
  [-0.235, 0.19, 0.005, 0.022], [-0.235, 0.19, -0.027, 0.022],
];

interface Fish {
  p: Vector3;
  v: Vector3;
  q: Quaternion;
  phase: number;
  wander: Vector3;
  dart: number;
  turn: number;
  size: number;
}

export interface Flake {
  p: Vector3;
  v: Vector3;
  state: 0 | 1 | 2; // floating, sinking, resting
  age: number;
  alive: boolean;
}

export class Life {
  root = new Group();
  fish: Fish[] = [];
  flakes: Flake[] = [];
  private fishMeshes: InstancedMesh[] = [];
  private swim = new Float32Array(FISH * 3);
  private swimAttr = new InstancedBufferAttribute(this.swim, 3);
  private flakeMesh: InstancedMesh;
  private snails: { o: Object3D; dir: number; speed: number; side: number }[] = [];
  private bubbles: InstancedMesh;
  private bubbleState: { y: number; x: number; z: number; v: number; r: number; out: number }[] = [];
  private motes: Points;
  private moteVel: Float32Array;
  focusFish = 0;

  constructor(fauna: Object3D) {
    this.root.name = 'life';
    // --- fish
    this.swimAttr.setUsage(DynamicDrawUsage);
    const parts = ['fish_body', 'fish_fins', 'fish_eyes'];
    for (const name of parts) {
      let src: Mesh | null = null;
      fauna.traverse((o) => {
        if ((o as Mesh).isMesh && o.name === name) src = o as Mesh;
      });
      if (!src) continue;
      const s = src as Mesh;
      const matName = (s.material as Material).name;
      const mat = swimPatch(makeMaterial(matName));
      mat.name = matName + '_swim';
      const geo = s.geometry.clone();
      geo.setAttribute('aSwim', this.swimAttr);
      const im = new InstancedMesh(geo, mat, FISH);
      im.instanceMatrix.setUsage(DynamicDrawUsage);
      im.castShadow = name !== 'fish_fins';
      im.receiveShadow = true;
      im.frustumCulled = false;
      im.layers.enable(OCCLUDER_LAYER);
      if (name === 'fish_fins') im.renderOrder = 3;
      this.root.add(im);
      this.fishMeshes.push(im);
    }
    for (let i = 0; i < FISH; i++) {
      const p = new Vector3((R() - 0.5) * 0.3, 0.12 + R() * 0.1, (R() - 0.5) * 0.12);
      const v = new Vector3(R() - 0.5, 0, R() - 0.5).setLength(0.04);
      this.fish.push({ p, v, q: new Quaternion(), phase: R() * 6, wander: new Vector3(), dart: 0, turn: 0, size: 0.95 + R() * 0.25 });
    }
    // --- snails on the front glass and a leaf or two
    let snailSrc: Object3D | null = null;
    fauna.traverse((o) => {
      if (o.name === 'snail') snailSrc = o;
    });
    if (snailSrc) {
      const spots: [number, number, number][] = [
        [-0.12, 0.25, 1], [0.07, 0.262, 1], [0.19, 0.09, 1], [-0.22, 0.16, -1],
      ];
      for (const [x, y, side] of spots) {
        const o = (snailSrc as Object3D).clone();
        o.traverse((c) => {
          const m = c as Mesh;
          if (m.isMesh) m.material = materialFor(((m.material as Material).name));
        });
        // foot to the glass: snail +Y (shell up) points into the tank
        o.position.set(x, y, side * (TANK.hz - 0.0002));
        o.rotation.set(side > 0 ? -Math.PI / 2 : Math.PI / 2, 0, 0, 'XYZ');
        o.scale.setScalar(0.8 + R() * 0.5);
        this.root.add(o);
        this.snails.push({ o, dir: R() * Math.PI * 2, speed: 0.0008 + R() * 0.0006, side });
      }
    }
    // --- food flakes
    const flakeGeo = new CircleGeometry(0.0022, 5);
    flakeGeo.rotateX(-Math.PI / 2);
    const flakeMat = patch(new MeshStandardMaterial({ roughness: 0.7, side: 2 }), { caustics: true });
    this.flakeMesh = new InstancedMesh(flakeGeo, flakeMat, 80);
    this.flakeMesh.count = 0;
    this.flakeMesh.frustumCulled = false;
    const palette = [0xc8402c, 0xd88a2a, 0x7a9a3a, 0xc9b25a, 0x8a3324];
    for (let i = 0; i < 80; i++) this.flakeMesh.setColorAt(i, new Color(palette[i % palette.length]));
    this.root.add(this.flakeMesh);
    // --- bubbles from the sponge filter's lift tube
    const bubMat = patch(
      new MeshStandardMaterial({ color: 0xffffff, roughness: 0.02, metalness: 1, transparent: true, opacity: 0.75, envMapIntensity: 2.5 }),
      { caustics: false },
    );
    this.bubbles = new InstancedMesh(new SphereGeometry(1, 10, 8), bubMat, 40);
    this.bubbles.instanceMatrix.setUsage(DynamicDrawUsage);
    this.bubbles.frustumCulled = false;
    for (let i = 0; i < 40; i++) {
      this.bubbleState.push({ y: 0.1 + R() * 0.17, x: 0, z: 0, v: 0.08 + R() * 0.05, r: 0.0012 + R() * 0.0018, out: 0 });
    }
    this.root.add(this.bubbles);
    // --- suspended particles (the specks visible in every photo)
    const n = 900;
    const pos = new Float32Array(n * 3);
    this.moteVel = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 3] = (R() - 0.5) * TANK.hx * 2;
      pos[i * 3 + 1] = 0.04 + R() * (WATER_Y - 0.05);
      pos[i * 3 + 2] = (R() - 0.5) * TANK.hz * 2;
    }
    const pg = new BufferGeometry();
    pg.setAttribute('position', new BufferAttribute(pos, 3).setUsage(DynamicDrawUsage));
    const pm = patch(
      new PointsMaterial({ color: 0xe8e0c8, size: 0.0011, sizeAttenuation: true, transparent: true, opacity: 0.55, blending: AdditiveBlending, depthWrite: false }),
      {},
    );
    this.motes = new Points(pg, pm);
    this.motes.frustumCulled = false;
    this.root.add(this.motes);
  }

  /** Drop a pinch of flakes at the surface around (x, z). */
  feed(x: number, z: number): void {
    for (let k = 0; k < 14; k++) {
      const p = new Vector3(
        Math.max(-TANK.hx + 0.01, Math.min(TANK.hx - 0.01, x + (R() - 0.5) * 0.04)),
        WATER_Y + 0.0006,
        Math.max(-TANK.hz + 0.01, Math.min(TANK.hz - 0.01, z + (R() - 0.5) * 0.03)),
      );
      this.flakes.push({ p, v: new Vector3((R() - 0.5) * 0.004, 0, (R() - 0.5) * 0.004), state: 0, age: R() * 2, alive: true });
    }
    if (this.flakes.length > 80) this.flakes.splice(0, this.flakes.length - 80);
  }

  update(dt: number, t: number): void {
    dt = Math.min(dt, 1 / 20);
    this.updateFlakes(dt);
    this.updateFish(dt, t);
    this.updateSnails(dt);
    this.updateBubbles(dt, t);
    this.updateMotes(dt, t);
  }

  private updateFlakes(dt: number): void {
    const m = new Matrix4(), q = new Quaternion(), s = new Vector3(1, 1, 1);
    let n = 0;
    for (const f of this.flakes) {
      if (!f.alive) continue;
      f.age += dt;
      if (f.state === 0) {
        f.p.addScaledVector(f.v, dt);
        if (f.age > 4) f.state = 1;
      } else if (f.state === 1) {
        f.p.y -= 0.007 * dt;
        f.p.x += Math.sin(f.age * 2.3 + f.p.z * 100) * 0.003 * dt;
        f.p.z += Math.cos(f.age * 1.9 + f.p.x * 100) * 0.003 * dt;
        const floor = surfaceY(f.p.x, f.p.z) + 0.0005;
        if (f.p.y < floor) {
          f.p.y = floor;
          f.state = 2;
        }
      } else if (f.age > 60) f.alive = false;
      q.setFromAxisAngle(new Vector3(Math.sin(f.age), 0.3, Math.cos(f.age * 0.7)).normalize(), f.state === 1 ? f.age * 2 : 0.2);
      m.compose(f.p, q, s);
      this.flakeMesh.setMatrixAt(n, m);
      n++;
    }
    this.flakes = this.flakes.filter((f) => f.alive);
    this.flakeMesh.count = n;
    this.flakeMesh.instanceMatrix.needsUpdate = true;
  }

  private updateFish(dt: number, t: number): void {
    const acc = new Vector3(), tmp = new Vector3(), sep = new Vector3(), ali = new Vector3(), coh = new Vector3();
    const m = new Matrix4(), s = new Vector3(), look = new Matrix4(), up = new Vector3(0, 1, 0), tq = new Quaternion();
    const minB = new Vector3(-TANK.hx + 0.03, 0.065, -TANK.hz + 0.03);
    const maxB = new Vector3(TANK.hx - 0.03, WATER_Y - 0.035, TANK.hz - 0.025);
    const food = this.flakes.filter((f) => f.state < 2 || f.age < 40);
    this.fish.forEach((f, i) => {
      acc.set(0, 0, 0);
      sep.set(0, 0, 0);
      ali.set(0, 0, 0);
      coh.set(0, 0, 0);
      let na = 0;
      for (let j = 0; j < this.fish.length; j++) {
        if (j === i) continue;
        const o = this.fish[j];
        tmp.subVectors(f.p, o.p);
        const d = tmp.length();
        if (d < 0.028) sep.addScaledVector(tmp, (0.028 - d) / (d + 1e-4));
        if (d < 0.09) {
          ali.add(o.v);
          coh.add(o.p);
          na++;
        }
      }
      acc.addScaledVector(sep, 0.9);
      if (na) {
        ali.divideScalar(na).sub(f.v);
        coh.divideScalar(na).sub(f.p);
        acc.addScaledVector(ali, 0.35).addScaledVector(coh, 0.08);
      }
      // wander: smooth random heading changes, mostly horizontal
      f.wander.x += (R() - 0.5) * dt * 0.6;
      f.wander.y += (R() - 0.5) * dt * 0.25;
      f.wander.z += (R() - 0.5) * dt * 0.6;
      f.wander.multiplyScalar(0.985);
      acc.add(f.wander);
      // food
      let target: Flake | null = null;
      let best = 0.22;
      for (const fl of food) {
        const d = fl.p.distanceTo(f.p);
        if (d < best) {
          best = d;
          target = fl;
        }
      }
      let maxSpeed = 0.055;
      if (target) {
        tmp.subVectors(target.p, f.p);
        acc.addScaledVector(tmp.normalize(), 0.5);
        maxSpeed = 0.15;
        if (best < 0.006) {
          target.alive = false;
          f.dart = 0.3;
        }
      }
      // occasional darts, the way white clouds dash after each other
      if (f.dart <= 0 && R() < dt * 0.05) f.dart = 0.4 + R() * 0.3;
      if (f.dart > 0) {
        f.dart -= dt;
        maxSpeed = Math.max(maxSpeed, 0.14);
        acc.addScaledVector(tmp.copy(f.v).normalize(), 0.4);
      }
      // walls (soft) and obstacles
      const wall = (v: number, lo: number, hi: number) => (v < lo ? (lo - v) * 30 : v > hi ? (hi - v) * 30 : 0);
      acc.x += wall(f.p.x, minB.x, maxB.x);
      acc.y += wall(f.p.y, minB.y, target ? WATER_Y - 0.004 : maxB.y);
      acc.z += wall(f.p.z, minB.z, maxB.z);
      for (const [ox, oy, oz, rr] of OBSTACLES) {
        tmp.set(f.p.x - ox, f.p.y - oy, f.p.z - oz);
        const d = tmp.length();
        const reach = rr + 0.02;
        if (d < reach) acc.addScaledVector(tmp, ((reach - d) / reach) * 3 / (d + 1e-4));
      }
      // keep level: damp vertical speed
      acc.y -= f.v.y * 1.2;
      const before = tmp.copy(f.v).normalize().clone();
      f.v.addScaledVector(acc, dt);
      const sp = f.v.length();
      const minSpeed = 0.012;
      if (sp > maxSpeed) f.v.multiplyScalar(maxSpeed / sp);
      if (sp < minSpeed) f.v.multiplyScalar(minSpeed / (sp + 1e-6));
      f.p.addScaledVector(f.v, dt);
      // orientation: face velocity, limited pitch
      const dir = tmp.copy(f.v).normalize();
      dir.y = Math.max(-0.6, Math.min(0.6, dir.y));
      dir.normalize();
      const turn = before.cross(dir).y / Math.max(dt, 1e-3);
      f.turn += (Math.max(-1, Math.min(1, turn * 0.25)) - f.turn) * Math.min(1, dt * 4);
      look.lookAt(new Vector3(), dir.clone().negate(), up);
      tq.setFromRotationMatrix(look);
      f.q.slerp(tq, Math.min(1, dt * 6));
      const speed = f.v.length();
      f.phase += dt * (9 + speed * 140);
      this.swim[i * 3] = f.phase;
      this.swim[i * 3 + 1] = 0.0007 + speed * 0.022;
      this.swim[i * 3 + 2] = -f.turn;
      s.setScalar(f.size);
      m.compose(f.p, f.q, s);
      for (const im of this.fishMeshes) im.setMatrixAt(i, m);
    });
    for (const im of this.fishMeshes) im.instanceMatrix.needsUpdate = true;
    this.swimAttr.needsUpdate = true;
    void t;
  }

  private updateSnails(dt: number): void {
    for (const sn of this.snails) {
      sn.dir += (R() - 0.5) * dt * 0.6;
      const dx = Math.cos(sn.dir) * sn.speed * dt;
      const dy = Math.sin(sn.dir) * sn.speed * dt;
      sn.o.position.x = Math.max(-TANK.hx + 0.01, Math.min(TANK.hx - 0.01, sn.o.position.x + dx));
      sn.o.position.y = Math.max(0.06, Math.min(WATER_Y - 0.006, sn.o.position.y + dy));
      if (sn.o.position.y >= WATER_Y - 0.006 || sn.o.position.y <= 0.06) sn.dir = -sn.dir;
      // the shell's forward (+X) follows dir in the glass plane
      sn.o.rotation.y = sn.side > 0 ? -sn.dir : sn.dir;
    }
  }

  private updateBubbles(dt: number, t: number): void {
    const m = new Matrix4(), q = new Quaternion(), s = new Vector3(), p = new Vector3();
    const tx = -0.168, tz = -0.05; // lift tube (blender (x, y) = (-0.168, 0.05))
    this.bubbleState.forEach((b, i) => {
      if (b.out <= 0) {
        b.y += b.v * dt;
        b.x = tx + Math.sin(t * 9 + i) * 0.0025;
        b.z = tz + Math.cos(t * 7 + i * 1.3) * 0.0025;
        if (b.y > 0.262) {
          b.out = 0.001;
          b.x = tx + 0.012;
          b.z = tz + 0.008;
          b.y = 0.27;
        }
      } else {
        // pushed out of the spout, popping at the surface
        b.out += dt;
        b.x += 0.05 * dt;
        b.y = Math.min(WATER_Y, b.y + 0.05 * dt);
        if (b.out > 0.35 + (i % 5) * 0.06) {
          b.out = 0;
          b.y = 0.1 + R() * 0.02;
        }
      }
      const wob = 1 + Math.sin(t * 30 + i) * 0.12;
      s.set(b.r * wob, b.r / wob, b.r * wob);
      p.set(b.x, b.y, b.z);
      m.compose(p, q, s);
      this.bubbles.setMatrixAt(i, m);
    });
    this.bubbles.instanceMatrix.needsUpdate = true;
  }

  private updateMotes(dt: number, t: number): void {
    const a = this.motes.geometry.attributes.position as BufferAttribute;
    const arr = a.array as Float32Array;
    for (let i = 0; i < a.count; i++) {
      const x = arr[i * 3], y = arr[i * 3 + 1], z = arr[i * 3 + 2];
      // a slow gyre: the filters push water left to right along the top
      const vx = 0.004 * Math.sin(y * 20 + t * 0.1) + 0.0015 * Math.sin(t * 0.3 + i);
      const vy = 0.0012 * Math.sin(x * 25 + t * 0.13 + i * 0.1) - 0.0003;
      const vz = 0.002 * Math.cos(x * 15 + y * 9 + t * 0.07);
      let nx = x + vx * dt, ny = y + vy * dt, nz = z + vz * dt;
      if (nx > TANK.hx) nx -= TANK.hx * 2;
      if (nx < -TANK.hx) nx += TANK.hx * 2;
      if (ny < 0.035) ny = WATER_Y - 0.01;
      if (ny > WATER_Y - 0.003) ny = 0.04;
      if (nz > TANK.hz) nz -= TANK.hz * 2;
      if (nz < -TANK.hz) nz += TANK.hz * 2;
      arr[i * 3] = nx;
      arr[i * 3 + 1] = ny;
      arr[i * 3 + 2] = nz;
    }
    a.needsUpdate = true;
    void this.moteVel;
  }

  fishPosition(i: number, out: Vector3): Vector3 {
    return out.copy(this.fish[i % this.fish.length].p);
  }
  fishVelocity(i: number, out: Vector3): Vector3 {
    return out.copy(this.fish[i % this.fish.length].v);
  }
}
