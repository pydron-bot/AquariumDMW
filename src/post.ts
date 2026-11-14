import {
  Camera,
  DepthTexture,
  FloatType,
  HalfFloatType,
  Matrix4,
  MeshBasicMaterial,
  OrthographicCamera,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  UnsignedIntType,
  Vector2,
  Vector3,
  WebGLRenderTarget,
  type WebGLRenderer,
} from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { FullScreenQuad, Pass } from 'three/examples/jsm/postprocessing/Pass.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { TANK, WATER_Y } from './tank/dims';
import { U } from './tank/shaders';

/** Objects on this layer block light for the shafts' depth map. */
export const OCCLUDER_LAYER = 2;

const FULLSCREEN_VS = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

/**
 * Volumetric light shafts through the water. Ray-marches each pixel's view
 * ray inside the water volume (half resolution), testing visibility against
 * a depth map rendered from the light, so the floating plants cut the beams
 * into shafts; the surface caustic pattern projected down each beam makes
 * them shimmer. Blurred, then added over the scene.
 */
class ShaftsPass extends Pass {
  lightCam = new OrthographicCamera(-0.33, 0.33, 0.24, -0.24, 0.05, 2.0);
  lightRT = new WebGLRenderTarget(768, 768);
  rtA = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
  rtB = new WebGLRenderTarget(1, 1, { type: HalfFloatType });
  steps = 40;
  enabledShafts = true;
  depthRef: { tex: DepthTexture | null };
  private march: ShaderMaterial;
  private blur: ShaderMaterial;
  private comp: ShaderMaterial;
  private quad = new FullScreenQuad();
  private frame = 0;
  private lightVP = new Matrix4();

  constructor(private scene: Scene, private camera: PerspectiveCamera, depthRef: { tex: DepthTexture | null }) {
    super();
    this.depthRef = depthRef;
    this.lightRT.depthTexture = new DepthTexture(768, 768, FloatType);
    this.lightCam.layers.set(OCCLUDER_LAYER);
    this.march = new ShaderMaterial({
      vertexShader: FULLSCREEN_VS,
      uniforms: {
        tDepth: { value: null },
        tLightDepth: { value: this.lightRT.depthTexture },
        uProjInv: { value: new Matrix4() },
        uViewInv: { value: new Matrix4() },
        uLightVP: { value: this.lightVP },
        uCamPos: { value: new Vector3() },
        uBoxMin: { value: new Vector3(-TANK.hx, 0, -TANK.hz) },
        uBoxMax: { value: new Vector3(TANK.hx, WATER_Y, TANK.hz) },
        uSteps: { value: 40 },
        uFrame: { value: 0 },
        uDensity: { value: 1.0 },
        uColor: { value: U.uLightColor.value },
        uLightDir: U.uLightDir,
        uTime: U.uTime,
        uAbsorb: U.uAbsorb,
        uWaterY: U.uWaterY,
        uCaustic: U.uCaustic,
      },
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        uniform sampler2D tDepth, tLightDepth;
        uniform mat4 uProjInv, uViewInv, uLightVP;
        uniform vec3 uCamPos, uBoxMin, uBoxMax, uColor, uLightDir, uAbsorb;
        uniform float uSteps, uFrame, uDensity, uTime, uWaterY, uCaustic;

        float caus(vec2 p, float t) {
          vec2 i = p; float c = 1.0;
          for (int n = 0; n < 3; n++) {
            float tt = t * (1.0 - (3.5 / float(n + 1)));
            i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
            c += 1.0 / length(vec2(p.x / (sin(i.x + tt) / 0.005), p.y / (cos(i.y + tt) / 0.005)));
          }
          c /= 3.0; c = 1.17 - pow(c, 1.4);
          return clamp(pow(abs(c), 5.0), 0.0, 2.0);
        }
        float ign(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }

        void main() {
          float d = texture2D(tDepth, vUv).r;
          vec4 v = uProjInv * vec4(vUv * 2.0 - 1.0, d * 2.0 - 1.0, 1.0);
          v /= v.w;
          vec3 wp = (uViewInv * v).xyz;
          vec3 rd = wp - uCamPos;
          float L = length(rd); rd /= L;
          vec3 inv = 1.0 / rd;
          vec3 t0 = (uBoxMin - uCamPos) * inv, t1 = (uBoxMax - uCamPos) * inv;
          vec3 mn = min(t0, t1), mx = max(t0, t1);
          float tn = max(max(max(mn.x, mn.y), mn.z), 0.0);
          float tf = min(min(min(mx.x, mx.y), mx.z), L);
          if (tf <= tn) { gl_FragColor = vec4(0.0); return; }
          float stepL = (tf - tn) / uSteps;
          float j = ign(gl_FragCoord.xy + uFrame * 5.588238);
          vec3 acc = vec3(0.0);
          float t = tn + stepL * j;
          for (int k = 0; k < 64; k++) {
            if (float(k) >= uSteps) break;
            vec3 p = uCamPos + rd * t;
            vec4 lc = uLightVP * vec4(p, 1.0);
            vec3 ln = lc.xyz / lc.w * 0.5 + 0.5;
            float ld = texture2D(tLightDepth, ln.xy).r;
            float lit = ln.z - 0.002 <= ld ? 1.0 : 0.0;
            float depth = uWaterY - p.y;
            vec2 s = p.xz + uLightDir.xz / max(uLightDir.y, 0.2) * depth;
            float c = mix(1.0, 0.25 + 2.2 * caus(mod(s * 95.0, 6.2831853) - 250.0, uTime * 0.55 + 23.0), uCaustic * 0.85);
            vec3 att = exp(-uAbsorb * (depth + (t - tn)));
            acc += lit * c * att;
            t += stepL;
          }
          // Forward scattering: brighter when looking up into the light.
          float g = 0.55;
          float cosT = dot(rd, uLightDir);
          float hg = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosT, 1.5) * 0.25;
          vec3 res = acc * stepL * uDensity * hg * uColor;
          // never let a bad sample (NaN/inf) reach the blur and bloom
          if (any(isnan(res)) || any(isinf(res))) res = vec3(0.0);
          gl_FragColor = vec4(res, 1.0);
        }`,
    });
    this.blur = new ShaderMaterial({
      vertexShader: FULLSCREEN_VS,
      uniforms: { tIn: { value: null }, uDir: { value: new Vector2() } },
      fragmentShader: /* glsl */ `
        varying vec2 vUv; uniform sampler2D tIn; uniform vec2 uDir;
        void main() {
          vec3 c = texture2D(tIn, vUv).rgb * 0.227;
          c += texture2D(tIn, vUv + uDir * 1.385).rgb * 0.316;
          c += texture2D(tIn, vUv - uDir * 1.385).rgb * 0.316;
          c += texture2D(tIn, vUv + uDir * 3.231).rgb * 0.070;
          c += texture2D(tIn, vUv - uDir * 3.231).rgb * 0.070;
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    this.comp = new ShaderMaterial({
      vertexShader: FULLSCREEN_VS,
      uniforms: { tDiffuse: { value: null }, tShafts: { value: null }, uOn: { value: 1 } },
      fragmentShader: /* glsl */ `
        varying vec2 vUv; uniform sampler2D tDiffuse, tShafts; uniform float uOn;
        void main() {
          vec4 c = texture2D(tDiffuse, vUv);
          gl_FragColor = vec4(c.rgb + texture2D(tShafts, vUv).rgb * uOn, c.a);
        }`,
    });
  }

  setSize(w: number, h: number): void {
    const hw = Math.max(1, Math.floor(w / 2)), hh = Math.max(1, Math.floor(h / 2));
    this.rtA.setSize(hw, hh);
    this.rtB.setSize(hw, hh);
  }

  updateLight(dir: Vector3): void {
    const c = new Vector3(0, 0.15, 0);
    this.lightCam.position.copy(c).addScaledVector(dir, 0.8);
    this.lightCam.lookAt(c);
    this.lightCam.updateMatrixWorld();
    this.lightCam.updateProjectionMatrix();
  }

  render(renderer: WebGLRenderer, writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget): void {
    this.depthRef.tex = readBuffer.depthTexture;
    if (!this.enabledShafts) {
      this.comp.uniforms.uOn.value = 0;
    } else {
      this.comp.uniforms.uOn.value = 1;
      // 1: depth from the light (occluders only)
      this.updateLight(U.uLightDir.value);
      this.lightVP.multiplyMatrices(this.lightCam.projectionMatrix, this.lightCam.matrixWorldInverse);
      const bg = this.scene.background;
      this.scene.background = null;
      const ov = this.scene.overrideMaterial;
      this.scene.overrideMaterial = depthOnly;
      renderer.setRenderTarget(this.lightRT);
      renderer.clear();
      renderer.render(this.scene, this.lightCam);
      this.scene.overrideMaterial = ov;
      this.scene.background = bg;
      // 2: march at half res
      const u = this.march.uniforms;
      u.tDepth.value = readBuffer.depthTexture;
      u.uProjInv.value.copy(this.camera.projectionMatrixInverse);
      u.uViewInv.value.copy(this.camera.matrixWorld);
      u.uCamPos.value.setFromMatrixPosition(this.camera.matrixWorld);
      u.uSteps.value = this.steps;
      u.uFrame.value = this.frame++ % 64;
      this.quad.material = this.march;
      renderer.setRenderTarget(this.rtA);
      this.quad.render(renderer);
      // 3: blur
      this.quad.material = this.blur;
      for (const [src, dst, dx, dy] of [
        [this.rtA, this.rtB, 1, 0],
        [this.rtB, this.rtA, 0, 1],
      ] as const) {
        this.blur.uniforms.tIn.value = src.texture;
        this.blur.uniforms.uDir.value.set(dx / src.width, dy / src.height);
        renderer.setRenderTarget(dst);
        this.quad.render(renderer);
      }
    }
    // 4: composite
    this.comp.uniforms.tDiffuse.value = readBuffer.texture;
    this.comp.uniforms.tShafts.value = this.rtA.texture;
    this.quad.material = this.comp;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.quad.render(renderer);
  }
}

const depthOnly = new MeshBasicMaterial({ colorWrite: false });

/** Depth of field (gather, golden-angle spiral), slight chromatic
 * aberration toward the edges, vignette and grain. Linear HDR in. */
class FinalPass extends Pass {
  mat: ShaderMaterial;
  private quad = new FullScreenQuad();
  constructor(private camera: PerspectiveCamera, private depthRef: { tex: DepthTexture | null }) {
    super();
    this.mat = new ShaderMaterial({
      vertexShader: FULLSCREEN_VS,
      uniforms: {
        tDiffuse: { value: null },
        tDepth: { value: null },
        uNear: { value: 0.01 },
        uFar: { value: 20 },
        uFocus: { value: 0.6 },
        uAperture: { value: 0.012 },
        uMaxBlur: { value: 9 },
        uRes: { value: new Vector2(1, 1) },
        uDof: { value: 1 },
        uTime: U.uTime,
        uVignette: { value: 0.32 },
        uGrain: { value: 0.025 },
        uCA: { value: 0.0018 },
      },
      fragmentShader: /* glsl */ `
        varying vec2 vUv;
        uniform sampler2D tDiffuse, tDepth;
        uniform float uNear, uFar, uFocus, uAperture, uMaxBlur, uDof, uTime, uVignette, uGrain, uCA;
        uniform vec2 uRes;
        float lin(float d) { float z = d * 2.0 - 1.0; return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear)); }
        float coc(float d) { float z = lin(d); return clamp(abs(z - uFocus) / max(z, 1e-3) * uAperture * uRes.y, 0.0, uMaxBlur); }
        vec3 tap(vec2 uv) {
          vec2 off = (uv - 0.5) * uCA;
          return vec3(texture2D(tDiffuse, uv - off).r, texture2D(tDiffuse, uv).g, texture2D(tDiffuse, uv + off).b);
        }
        float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
        void main() {
          vec3 col;
          float c0 = coc(texture2D(tDepth, vUv).r) * uDof;
          if (c0 < 0.6) {
            col = tap(vUv);
          } else {
            vec3 acc = tap(vUv); float wsum = 1.0;
            const float GA = 2.39996323;
            for (int i = 1; i < 28; i++) {
              float r = sqrt(float(i) / 28.0) * c0;
              float a = float(i) * GA;
              vec2 uv = vUv + vec2(cos(a), sin(a)) * r / uRes;
              float cs = coc(texture2D(tDepth, uv).r) * uDof;
              // a sample only bleeds in if it is itself blurred that far (keeps sharp edges sharp)
              float w = smoothstep(r - 1.0, r + 1.0, max(cs, c0 * 0.6));
              acc += tap(uv) * w; wsum += w;
            }
            col = acc / wsum;
          }
          vec2 q = vUv - 0.5;
          col *= 1.0 - uVignette * dot(q, q) * 2.2;
          col += (hash(vUv * uRes + fract(uTime) * 91.7) - 0.5) * uGrain * (0.4 + col);
          gl_FragColor = vec4(col, 1.0);
        }`,
    });
  }
  setSize(w: number, h: number): void {
    this.mat.uniforms.uRes.value.set(w, h);
  }
  render(renderer: WebGLRenderer, writeBuffer: WebGLRenderTarget, readBuffer: WebGLRenderTarget): void {
    const u = this.mat.uniforms;
    u.tDiffuse.value = readBuffer.texture;
    u.tDepth.value = this.depthRef.tex;
    u.uNear.value = this.camera.near;
    u.uFar.value = this.camera.far;
    this.quad.material = this.mat;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.quad.render(renderer);
  }
}

export interface Post {
  composer: EffectComposer;
  shafts: ShaftsPass;
  bloom: UnrealBloomPass;
  final: FinalPass;
  setSize(w: number, h: number, dpr: number): void;
}

export function buildPost(renderer: WebGLRenderer, scene: Scene, camera: PerspectiveCamera, msaa: number): Post {
  const size = renderer.getDrawingBufferSize(new Vector2());
  const rt = new WebGLRenderTarget(size.x, size.y, { type: HalfFloatType, samples: msaa });
  rt.depthTexture = new DepthTexture(size.x, size.y, UnsignedIntType);
  const composer = new EffectComposer(renderer, rt);
  const depthRef: { tex: DepthTexture | null } = { tex: null };
  composer.addPass(new RenderPass(scene, camera as Camera));
  const shafts = new ShaftsPass(scene, camera, depthRef);
  composer.addPass(shafts);
  const bloom = new UnrealBloomPass(new Vector2(size.x / 2, size.y / 2), 0.55, 0.5, 1.6);
  composer.addPass(bloom);
  const final = new FinalPass(camera, depthRef);
  composer.addPass(final);
  composer.addPass(new OutputPass());
  return {
    composer,
    shafts,
    bloom,
    final,
    setSize(w, h, dpr) {
      composer.setPixelRatio(dpr);
      composer.setSize(w, h);
    },
  };
}
