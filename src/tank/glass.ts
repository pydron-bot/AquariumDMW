import {
  AdditiveBlending,
  BoxGeometry,
  Color,
  DoubleSide,
  FrontSide,
  Group,
  Mesh,
  MeshPhysicalMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  ShaderMaterial,
  Vector2,
  type Texture,
} from 'three';
import { Reflector } from 'three/examples/jsm/objects/Reflector.js';
import { TANK, WATER_Y } from './dims';
import { tex } from './materials';
import { patch, U } from './shaders';

/**
 * Glass, condensation, lid and the water surface.
 * - Panes add reflections only (additive), so they never darken the view.
 * - The pane edges glow green-cyan like real float glass seen edge-on.
 * - Above the waterline the glass is fogged with drops, as in the photos.
 * - The surface seen from below is a mirror (total internal reflection)
 *   past the critical angle, via a Reflector, distorted by small waves.
 */
export interface Glass {
  root: Group;
  mirror: Reflector | null;
  setMirror(on: boolean): void;
}

export function buildGlass(withMirror: boolean): Glass {
  const root = new Group();
  root.name = 'glass';
  const { hx, hz, h, glass: g } = TANK;

  const paneMat = new MeshPhysicalMaterial({
    color: 0x000000,
    roughness: 0.03,
    metalness: 0,
    specularIntensity: 1,
    envMapIntensity: 1.4,
    blending: AdditiveBlending,
    transparent: true,
    depthWrite: false,
    side: FrontSide,
  });
  patch(paneMat, { water: false });
  const panes: [number, number, number, number, number, number][] = [
    // w, h, d, x, y, z
    [hx * 2 + g * 2, h, g, 0, h / 2, hz + g / 2],
    [hx * 2 + g * 2, h, g, 0, h / 2, -hz - g / 2],
    [g, h, hz * 2, -hx - g / 2, h / 2, 0],
    [g, h, hz * 2, hx + g / 2, h / 2, 0],
  ];
  for (const [w, hh, d, x, y, z] of panes) {
    const m = new Mesh(new BoxGeometry(w, hh, d), paneMat);
    m.position.set(x, y, z);
    m.renderOrder = 10;
    root.add(m);
  }

  // Edges: the green of iron in soda-lime glass, brightest edge-on.
  const edgeMat = new MeshStandardMaterial({
    color: 0x6fd8c4,
    emissive: new Color(0x2a8c7a),
    emissiveIntensity: 0.35,
    roughness: 0.1,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      const e = new Mesh(new BoxGeometry(g * 1.1, h - 0.03, g * 1.1), edgeMat);
      e.position.set(sx * (hx + g / 2), h / 2 - 0.004, sz * (hz + g / 2));
      e.renderOrder = 11;
      root.add(e);
    }
  }

  // Condensation band between the waterline and the rim, and on the lid.
  const drops = tex('drops.png', false, true);
  const dropMat = (repeat: Vector2): MeshStandardMaterial => {
    const t = drops.clone();
    t.repeat.copy(repeat);
    const m = new MeshStandardMaterial({
      color: 0xdfe8f0,
      roughness: 0.12,
      normalMap: t,
      alphaMap: t as Texture,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      side: DoubleSide,
    });
    m.normalScale.set(1.5, 1.5);
    // Drop coverage lives in the normal map's alpha; alphaMap reads .g, so
    // swap in .a, and keep the fogged film faint between drops.
    m.onBeforeCompile = (s) => {
      s.fragmentShader = s.fragmentShader.replace(
        '#include <alphamap_fragment>',
        'diffuseColor.a *= 0.08 + 0.55 * texture2D( alphaMap, vAlphaMapUv ).a;',
      );
    };
    return patch(m, { water: false });
  };
  const bandH = h - 0.016 - WATER_Y;
  const bandY = WATER_Y + bandH / 2;
  const front = new Mesh(new PlaneGeometry(hx * 2, bandH), dropMat(new Vector2(6, 0.36)));
  front.position.set(0, bandY, hz - 0.0003);
  front.renderOrder = 9;
  root.add(front);
  const back = front.clone();
  back.position.z = -hz + 0.0003;
  root.add(back);
  for (const sx of [-1, 1]) {
    const s = new Mesh(new PlaneGeometry(hz * 2, bandH), dropMat(new Vector2(3, 0.36)));
    s.rotation.y = Math.PI / 2;
    s.position.set(sx * (hx - 0.0003), bandY, 0);
    s.renderOrder = 9;
    root.add(s);
  }
  // Lid: clear glass (reflections) plus the drops on its underside.
  const lid = new Mesh(new BoxGeometry(hx * 2 - 0.02, 0.003, hz * 2 - 0.04), paneMat);
  lid.position.set(0, h - 0.0025, -0.005);
  lid.renderOrder = 10;
  root.add(lid);
  const lidDrops = new Mesh(new PlaneGeometry(hx * 2 - 0.02, hz * 2 - 0.04), dropMat(new Vector2(6, 3)));
  lidDrops.rotation.x = Math.PI / 2;
  lidDrops.position.set(0, h - 0.0045, -0.005);
  lidDrops.renderOrder = 9;
  root.add(lidDrops);

  // Water surface seen from above: clear, slightly rippled, reflective.
  const surfTop = new MeshPhysicalMaterial({
    color: 0x0,
    roughness: 0.04,
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
    envMapIntensity: 1.2,
  });
  surfTop.onBeforeCompile = (s) => {
    s.fragmentShader = s.fragmentShader.replace(
        '#include <normal_fragment_maps>',
        `#include <normal_fragment_maps>
        {
          vec2 p = vTkWorld.xz * 100.0;
          vec2 dn = vec2(cos(p.x * 1.7 + uTime * 1.3) + 0.6 * cos(p.x * 3.1 - p.y * 2.3 + uTime * 2.1),
                         cos(p.y * 2.3 - uTime * 1.1) + 0.6 * cos(p.y * 3.7 + p.x * 1.1 + uTime * 1.7)) * 0.04;
          normal = normalize(normal + vec3(dn.x, dn.y, 0.0));
        }`,
      );
  };
  patch(surfTop, { water: false });
  const top = new Mesh(new PlaneGeometry(hx * 2, hz * 2), surfTop);
  top.rotation.x = -Math.PI / 2;
  top.position.y = WATER_Y;
  top.renderOrder = 8;
  root.add(top);

  // From below: total internal reflection band.
  let mirror: Reflector | null = null;
  if (withMirror) {
    mirror = new Reflector(new PlaneGeometry(hx * 2, hz * 2), {
      textureWidth: 1024,
      textureHeight: 512,
      clipBias: 0.001,
      shader: TIR_SHADER,
    });
    mirror.rotation.x = Math.PI / 2; // faces down into the water
    mirror.position.y = WATER_Y - 0.0004;
    mirror.renderOrder = 7;
    const mat = mirror.material as ShaderMaterial;
    mat.transparent = true;
    mat.depthWrite = false;
    mat.uniforms.uTime = U.uTime;
    root.add(mirror);
  }

  return {
    root,
    mirror,
    setMirror(on: boolean) {
      if (mirror) mirror.visible = on;
    },
  };
}

/** Reflector shader: mirror strength by the Fresnel/TIR curve for water to
 * air (critical angle ~48.6°), slightly wavy. */
const TIR_SHADER = {
  name: 'TIRShader',
  uniforms: {
    color: { value: null },
    tDiffuse: { value: null },
    textureMatrix: { value: null },
    uTime: { value: 0 },
  },
  vertexShader: /* glsl */ `
    uniform mat4 textureMatrix;
    varying vec4 vUv;
    varying vec3 vWorld;
    void main() {
      vUv = textureMatrix * vec4(position, 1.0);
      vec4 w = modelMatrix * vec4(position, 1.0);
      vWorld = w.xyz;
      gl_Position = projectionMatrix * viewMatrix * w;
    }`,
  fragmentShader: /* glsl */ `
    uniform vec3 color;
    uniform sampler2D tDiffuse;
    uniform float uTime;
    varying vec4 vUv;
    varying vec3 vWorld;
    void main() {
      vec3 v = normalize(cameraPosition - vWorld);
      float cosI = clamp(-v.y, 0.0, 1.0);           // angle to the downward normal
      float sinI = sqrt(1.0 - cosI * cosI);
      float tir = smoothstep(0.70, 0.78, sinI);      // sin(48.6°) = 0.75
      float fres = 0.02 + 0.98 * pow(1.0 - cosI, 5.0);
      float k = max(tir, fres);
      vec2 w = vec2(sin(vWorld.x * 140.0 + uTime * 1.7) + sin(vWorld.z * 210.0 - uTime * 1.3),
                    cos(vWorld.z * 160.0 + uTime * 1.1) + sin(vWorld.x * 90.0 + uTime * 2.3)) * 0.004;
      vec4 uv = vUv; uv.xy += w * uv.w;
      vec3 c = texture2DProj(tDiffuse, uv).rgb;
      gl_FragColor = vec4(c * 0.92, k * 0.95);
    }`,
};
