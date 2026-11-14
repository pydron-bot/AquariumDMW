import {
  BoxGeometry,
  Color,
  DirectionalLight,
  Group,
  HemisphereLight,
  Mesh,
  MeshStandardMaterial,
  PlaneGeometry,
  PMREMGenerator,
  Scene,
  type WebGLRenderer,
} from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { tex } from './materials';
import { patch, U } from './shaders';

/** The room around the tank: back wall washed purple by the RGB bar, the
 * dark woodgrain stand, an environment for reflections, and the lights. */
export interface Room {
  root: Group;
  sun: DirectionalLight;
  tint: DirectionalLight;
  hemi: HemisphereLight;
  wall: MeshStandardMaterial;
  setMode(mode: LightMode): void;
}

export type LightMode = 'rgb' | 'day' | 'moon';

export function buildRoom(renderer: WebGLRenderer, scene: Scene): Room {
  const root = new Group();
  root.name = 'room';

  const pmrem = new PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.22;

  // Back wall: a soft pool of purple light behind and above the tank.
  const wall = new MeshStandardMaterial({ color: 0xbdb6c4, roughness: 0.95 });
  wall.onBeforeCompile = (s) => {
    s.uniforms.uWash = { value: new Color(0.55, 0.28, 0.85) };
    s.uniforms.uWashK = wallWash;
    s.fragmentShader = s.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform vec3 uWash; uniform float uWashK;')
      .replace(
        '#include <opaque_fragment>',
        `{
          vec2 q = (vTkWorld.xy - vec2(0.0, 0.42)) / vec2(0.75, 0.42);
          float pool = exp(-dot(q, q) * 1.4);
          outgoingLight += uWash * pool * uWashK;
        }
        #include <opaque_fragment>`,
      );
  };
  patch(wall, {});
  const wallMesh = new Mesh(new PlaneGeometry(4, 2.4), wall);
  wallMesh.position.set(0, 0.6, -0.36);
  wallMesh.receiveShadow = true;
  root.add(wallMesh);

  // Stand: dark woodgrain top, the tank sits on it.
  const deskMat = patch(
    new MeshStandardMaterial({ map: tex('desk_color.jpg', true, true), roughness: 0.55, color: 0xd8d8d8 }),
    {},
  );
  deskMat.map!.repeat.set(2, 1.2);
  const desk = new Mesh(new BoxGeometry(1.25, 0.03, 0.62), deskMat);
  desk.position.set(0, -0.012 - 0.015, -0.03);
  desk.receiveShadow = true;
  root.add(desk);
  const shelf = new Mesh(new BoxGeometry(1.25, 0.7, 0.03), deskMat);
  shelf.position.set(0, -0.38, 0.27);
  root.add(shelf);

  // Main light: the bar's white LEDs, straight down through the lid.
  const sun = new DirectionalLight(0xf4f6ff, 3.4);
  sun.position.copy(U.uLightDir.value).multiplyScalar(0.8);
  sun.target.position.set(0, 0, 0);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = sun.shadow.camera;
  sc.left = -0.32;
  sc.right = 0.32;
  sc.top = 0.24;
  sc.bottom = -0.24;
  sc.near = 0.05;
  sc.far = 1.6;
  sun.shadow.bias = -0.0003;
  sun.shadow.normalBias = 0.0015;
  sun.shadow.radius = 3;
  root.add(sun, sun.target);

  // The bar's red/green/blue diodes: a purple-magenta fill from slightly
  // in front, no shadows.
  const tint = new DirectionalLight(0xa070ff, 0.8);
  tint.position.set(-0.2, 0.8, 0.5);
  root.add(tint);

  const hemi = new HemisphereLight(0x9a7cd0, 0x2a2030, 0.45);
  root.add(hemi);

  const room: Room = {
    root,
    sun,
    tint,
    hemi,
    wall,
    setMode(mode) {
      if (mode === 'rgb') {
        sun.color.set(0xf4f6ff);
        sun.intensity = 3.4;
        tint.color.set(0xa070ff);
        tint.intensity = 0.8;
        hemi.color.set(0x9a7cd0);
        hemi.intensity = 0.45;
        wallWash.value = 1;
        U.uScatter.value.setRGB(0.055, 0.075, 0.045);
      } else if (mode === 'day') {
        sun.color.set(0xfff4e6);
        sun.intensity = 4.2;
        tint.color.set(0xffffff);
        tint.intensity = 0.2;
        hemi.color.set(0xe8eef4);
        hemi.intensity = 0.6;
        wallWash.value = 0.12;
        U.uScatter.value.setRGB(0.07, 0.09, 0.06);
      } else {
        sun.color.set(0x4a6cff);
        sun.intensity = 0.9;
        tint.color.set(0x2030ff);
        tint.intensity = 0.25;
        hemi.color.set(0x1a2040);
        hemi.intensity = 0.15;
        wallWash.value = 0.25;
        U.uScatter.value.setRGB(0.01, 0.018, 0.04);
      }
      U.uLightColor.value.copy(sun.color).multiplyScalar(sun.intensity / 3.4);
    },
  };
  room.setMode('rgb');
  return room;
}

const wallWash = { value: 1 };
