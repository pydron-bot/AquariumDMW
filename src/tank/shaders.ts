import { Color, Material, ShaderChunk, Vector3, type WebGLProgramParametersWithUniforms } from 'three';
import { TANK, WATER_Y } from './dims';

/**
 * Shared uniforms for every patched material. One object, so a slider or a
 * light-mode change reaches all of them at once.
 */
export const U = {
  uTime: { value: 0 },
  uWaterY: { value: WATER_Y },
  uBoxMin: { value: new Vector3(-TANK.hx, 0, -TANK.hz) },
  uBoxMax: { value: new Vector3(TANK.hx, WATER_Y, TANK.hz) },
  /** Per-metre absorption of the (slightly tannin-stained) water. */
  uAbsorb: { value: new Vector3(1.35, 0.55, 1.05) },
  /** In-scattered haze colour (already scaled by the light). */
  uScatter: { value: new Color(0.055, 0.075, 0.045) },
  uScatterK: { value: 2.2 },
  uWaterOn: { value: 1 },
  uCaustic: { value: 1 },
  uCausticScale: { value: 95 },
  /** Direction toward the main light (world). */
  uLightDir: { value: new Vector3(0.05, 1, 0.12).normalize() },
  uLightColor: { value: new Color(1, 1, 1) },
  uSway: { value: 1 },
  uTransl: { value: 0.35 },
};

export interface PatchOptions {
  /** Absorb/scatter along the part of the view ray that's under water. */
  water?: boolean;
  /** Caustics on the main light, below the surface. */
  caustics?: boolean;
  /** Vertex sway: amplitude per metre² of height (plants), or hanging (roots). */
  sway?: number;
  swayMode?: 'plant' | 'hang' | 'float';
  /** Thin-leaf translucency: backlit leaves glow. */
  translucent?: boolean;
}

const PARS_VERTEX = /* glsl */ `
uniform float uTime;
uniform float uSwayAmt;
uniform float uSway;
varying vec3 vTkWorld;
`;

const SWAY_VERTEX = /* glsl */ `
#ifdef TK_SWAY
{
  vec4 tkB = vec4(0.0, 0.0, 0.0, 1.0);
  #ifdef USE_INSTANCING
  tkB = instanceMatrix * tkB;
  #endif
  tkB = modelMatrix * tkB;
  float ph = tkB.x * 13.0 + tkB.z * 7.0;
  #if TK_SWAY == 1
    float h = max(transformed.y, 0.0);
    float k = uSwayAmt * uSway * h * h;
    transformed.x += k * (sin(uTime * 0.8 + ph + h * 18.0) + 0.35 * sin(uTime * 2.1 + ph * 1.7 + h * 41.0));
    transformed.z += k * 0.7 * cos(uTime * 0.63 + ph * 1.3 + h * 15.0);
  #elif TK_SWAY == 2
    float h = max(-transformed.y, 0.0);
    float k = uSwayAmt * uSway * h * h;
    transformed.x += k * sin(uTime * 0.55 + ph + h * 4.0);
    transformed.z += k * cos(uTime * 0.47 + ph * 0.7 + h * 3.0);
  #else
    transformed.y += uSwayAmt * uSway * sin(uTime * 1.3 + ph * 3.0);
  #endif
}
#endif
`;

const WORLD_VERTEX = /* glsl */ `
{
  vec4 tkW = vec4(transformed, 1.0);
  #ifdef USE_INSTANCING
  tkW = instanceMatrix * tkW;
  #endif
  vTkWorld = (modelMatrix * tkW).xyz;
}
`;

const PARS_FRAGMENT = /* glsl */ `
uniform float uTime;
uniform float uWaterY;
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform vec3 uAbsorb;
uniform vec3 uScatter;
uniform float uScatterK;
uniform float uWaterOn;
uniform float uCaustic;
uniform float uCausticScale;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform float uTransl;
varying vec3 vTkWorld;

// Iterated-trig water caustic (after the well-known Shadertoy pattern).
float tkCausticF(vec2 p, float t) {
  vec2 i = p;
  float c = 1.0;
  float inten = 0.005;
  for (int n = 0; n < 4; n++) {
    float tt = t * (1.0 - (3.5 / float(n + 1)));
    i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
    c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / inten), p.y / (cos(i.y + tt) / inten)));
  }
  c /= 4.0;
  c = 1.17 - pow(c, 1.4);
  return clamp(pow(abs(c), 7.0), 0.0, 2.0);
}

vec3 tkCaustic(vec3 wp) {
  float depth = uWaterY - wp.y;
  if (depth < 0.0 || uCaustic <= 0.0) return vec3(1.0);
  // Project along the light to where this ray crossed the surface.
  vec2 s = wp.xz + uLightDir.xz / max(uLightDir.y, 0.2) * depth;
  // The pattern tiles on 2π and is tuned for coordinates near -250.
  vec2 p = mod(s * uCausticScale, 6.2831853) - 250.0;
  float t = uTime * 0.55 + 23.0;
  // Slight dispersion: the three channels focus at different depths.
  float spread = 0.015 + depth * 0.12;
  vec3 c = vec3(
    tkCausticF(p, t),
    tkCausticF(p + vec2(spread * 0.5), t),
    tkCausticF(p + vec2(spread), t));
  // Caustics blur with depth (and the light isn't a point).
  float sharp = mix(1.0, 0.45, clamp(depth * 3.5, 0.0, 1.0));
  return mix(vec3(1.0), vec3(0.62) + c * 2.2 * sharp, uCaustic);
}

float tkWaterPath(vec3 ro, vec3 target) {
  vec3 rd = target - ro;
  float L = length(rd);
  rd /= L;
  vec3 inv = 1.0 / rd;
  vec3 t0 = (uBoxMin - ro) * inv;
  vec3 t1 = (uBoxMax - ro) * inv;
  vec3 tmn = min(t0, t1);
  vec3 tmx = max(t0, t1);
  float tn = max(max(tmn.x, tmn.y), tmn.z);
  float tf = min(min(tmx.x, tmx.y), tmx.z);
  return max(0.0, min(tf, L) - max(tn, 0.0)) * uWaterOn;
}
`;

const CAUSTIC_LIGHT = /* glsl */ `
getDirectionalLightInfo( directionalLight, directLight );
#if defined(TK_CAUSTICS) && UNROLLED_LOOP_INDEX == 0
directLight.color *= tkCausticNow;
#endif
`;

const TRANSLUCENT = /* glsl */ `
#ifdef TK_TRANSL
{
  vec3 wn = normalize((vec4(normal, 0.0) * viewMatrix).xyz);
  vec3 vdir = normalize(cameraPosition - vTkWorld);
  // Light passing through a thin leaf toward the viewer.
  float back = max(0.0, -dot(wn, uLightDir) * sign(dot(wn, vdir)));
  float through = pow(max(0.0, dot(-vdir, uLightDir)), 2.0) * 0.6 + 0.4;
  outgoingLight += diffuseColor.rgb * uLightColor * uTransl * back * through * tkCausticNow;
}
#endif
`;

const WATER_FRAGMENT = /* glsl */ `
#ifdef TK_WATER
{
  float d = tkWaterPath(cameraPosition, vTkWorld);
  vec3 tr = exp(-uAbsorb * d);
  gl_FragColor.rgb = gl_FragColor.rgb * tr + uScatter * (1.0 - exp(-uScatterK * d));
}
#endif
#include <fog_fragment>
`;

/** Adds the tank effects to a built-in material (Standard/Physical/Basic). */
export function patch<T extends Material>(mat: T, opt: PatchOptions = {}): T {
  const defines: Record<string, string | number> = {};
  if (opt.water !== false) defines.TK_WATER = '';
  if (opt.caustics) defines.TK_CAUSTICS = '';
  if (opt.translucent) defines.TK_TRANSL = '';
  if (opt.sway) defines.TK_SWAY = opt.swayMode === 'hang' ? 2 : opt.swayMode === 'float' ? 3 : 1;
  const swayAmt = { value: opt.sway ?? 0 };
  mat.defines = { ...(mat.defines ?? {}), ...defines };
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader: WebGLProgramParametersWithUniforms, renderer) => {
    prev?.call(mat, shader, renderer);
    Object.assign(shader.uniforms, U, { uSwayAmt: swayAmt });
    shader.vertexShader = PARS_VERTEX + shader.vertexShader
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + SWAY_VERTEX)
      .replace('#include <project_vertex>', '#include <project_vertex>\n' + WORLD_VERTEX);
    let fs = PARS_FRAGMENT + shader.fragmentShader;
    const hasLights = fs.includes('#include <lights_fragment_begin>');
    if (hasLights) {
      fs = fs.replace(
        '#include <lights_fragment_begin>',
        '#ifdef TK_CAUSTICS\nvec3 tkCausticNow = tkCaustic(vTkWorld);\n#else\nvec3 tkCausticNow = vec3(1.0);\n#endif\n#include <lights_fragment_begin>',
      );
      fs = fs.replace('#include <opaque_fragment>', TRANSLUCENT + '\n#include <opaque_fragment>');
    }
    fs = fs.replace('#include <fog_fragment>', WATER_FRAGMENT);
    shader.fragmentShader = fs;
  };
  // Distinct programs per option set and per material-specific hook.
  const key = JSON.stringify(defines) + prev.toString();
  const prevKey = mat.customProgramCacheKey.bind(mat);
  mat.customProgramCacheKey = () => prevKey() + key;
  return mat;
}

/**
 * The caustic multiply sits inside the directional-light loop, after the
 * shadow term, so caustics are shadowed by the floating plants. three.js
 * resolves `#include` before onBeforeCompile sees the chunks only for the
 * top level, so the loop chunk itself is patched once, globally (the
 * #if keeps unpatched materials unchanged).
 */
const LOOP = 'getDirectionalLightInfo( directionalLight, directLight );';
if (!ShaderChunk.lights_fragment_begin.includes('TK_CAUSTICS')) {
  ShaderChunk.lights_fragment_begin = ShaderChunk.lights_fragment_begin.replace(LOOP, CAUSTIC_LIGHT);
}
