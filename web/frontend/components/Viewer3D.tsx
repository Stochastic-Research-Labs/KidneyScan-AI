"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Grid } from "../lib/nifti";
import { LABELS } from "../lib/labels";

/* -------------------------------------------------------------------------- */
/* Shaders: GPU ray-marching through the CT (and label) volume.                */
/* Box-local axes: x = patient Left, y = Superior, z = Anterior.               */
/* Texture axes:   s = Left, t = Anterior, r = Superior.                       */
/* -------------------------------------------------------------------------- */
const VERT = /* glsl */ `
out vec3 vOrigin;
out vec3 vDir;
void main() {
  vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
  vDir = position - vOrigin;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}`;

const FRAG = /* glsl */ `
precision highp float;
in vec3 vOrigin;
in vec3 vDir;
uniform highp sampler3D uCT;
uniform highp sampler3D uMask;
uniform float uMode;      // 0 = original CT, 1 = segmentation overlay
uniform float uHasCT;
uniform float uHasMask;
uniform float uDensity;
uniform float uClip;      // cuts the anterior side away to look inside
uniform vec3 uShow;       // kidney, tumor, cyst visibility
uniform int uSteps;
uniform vec3 uColK;
uniform vec3 uColT;
uniform vec3 uColC;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec2 hitBox(vec3 o, vec3 d) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(-0.5) - o) * inv;
  vec3 t1 = (vec3(0.5) - o) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
  vec3 rd = normalize(vDir);
  vec2 b = hitBox(vOrigin, rd);
  if (b.x > b.y) discard;
  b.x = max(b.x, 0.0);
  float dt = (b.y - b.x) / float(uSteps);
  vec3 p = vOrigin + rd * (b.x + dt * hash(gl_FragCoord.xy));
  vec3 acc = vec3(0.0);
  float al = 0.0;

  for (int i = 0; i < 400; i++) {
    if (i >= uSteps) break;
    if (p.z < 0.5 - uClip) {
      vec3 tc = vec3(p.x, p.z, p.y) + 0.5;
      float c = uHasCT > 0.5 ? texture(uCT, tc).r : 0.0;
      float d = smoothstep(0.14, 1.0, c);
      d *= d;
      float k = uDensity * (uMode < 0.5 ? 7.0 : 2.0);
      float a1 = 1.0 - exp(-k * d * dt);
      vec3 c1 = mix(vec3(0.46, 0.62, 0.78), vec3(1.0, 0.96, 0.88), smoothstep(0.35, 1.0, c));

      float a2 = 0.0;
      vec3 c2 = vec3(0.0);
      if (uMode > 0.5 && uHasMask > 0.5) {
        float lab = floor(texture(uMask, tc).r * 255.0 + 0.5);
        if (lab > 0.5 && lab < 1.5 && uShow.x > 0.5) { c2 = uColK; a2 = 1.0 - exp(-8.0 * dt); }
        else if (lab > 1.5 && lab < 2.5 && uShow.y > 0.5) { c2 = uColT; a2 = 1.0 - exp(-90.0 * dt); }
        else if (lab > 2.5 && lab < 3.5 && uShow.z > 0.5) { c2 = uColC; a2 = 1.0 - exp(-90.0 * dt); }
      }
      float a = a2 + (1.0 - a2) * a1;
      vec3 col = a > 1e-5 ? (c2 * a2 + c1 * a1 * (1.0 - a2)) / a : vec3(0.0);
      acc += (1.0 - al) * a * col;
      al += (1.0 - al) * a;
      if (al > 0.97) break;
    }
    p += rd * dt;
  }
  gl_FragColor = vec4(acc / max(al, 1e-4), al);
}`;

const hexToVec = (h: string) =>
  new THREE.Vector3(parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255);

/* -------------------------------------------------------------------------- */
/* One WebGL panel                                                             */
/* -------------------------------------------------------------------------- */
const HOME = new THREE.Vector3(0.9, 0.45, 2.3);

function emptyTexture() {
  const t = new THREE.Data3DTexture(new Uint8Array([0]), 1, 1, 1);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}

function gridTexture(g: Grid, nearest: boolean) {
  const t = new THREE.Data3DTexture(g.data as unknown as BufferSource, g.size[0], g.size[1], g.size[2]);
  t.format = THREE.RedFormat;
  t.type = THREE.UnsignedByteType;
  t.minFilter = t.magFilter = nearest ? THREE.NearestFilter : THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}

type Panel = ReturnType<typeof makePanel>;

function makePanel(host: HTMLElement, mode: 0 | 1, steps: number) {
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const canvas = renderer.domElement;
  canvas.style.cssText = "display:block;width:100%;height:100%;touch-action:none;outline:none;";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", mode === 0 ? "3D view of the original CT. Drag to rotate." : "3D view of the segmentation overlay. Drag to rotate.");
  host.appendChild(canvas);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 20);
  camera.position.copy(HOME);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.09;
  controls.enablePan = false;
  controls.minDistance = 1;
  controls.maxDistance = 5;
  controls.autoRotateSpeed = 2.4;

  const uniforms = {
    uCT: { value: emptyTexture() },
    uMask: { value: emptyTexture() },
    uMode: { value: mode },
    uHasCT: { value: 0 },
    uHasMask: { value: 0 },
    uDensity: { value: 1 },
    uClip: { value: 0 },
    uShow: { value: new THREE.Vector3(1, 1, 1) },
    uSteps: { value: steps },
    uColK: { value: hexToVec(LABELS[0].color) },
    uColT: { value: hexToVec(LABELS[1].color) },
    uColC: { value: hexToVec(LABELS[2].color) },
  };
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
  });
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  let dirty = true;
  controls.addEventListener("change", () => (dirty = true));

  return {
    camera,
    controls,
    uniforms,
    markDirty: () => (dirty = true),
    resize() {
      const w = host.clientWidth || 1;
      const h = host.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      dirty = true;
    },
    setGrids(ct: Grid | null, mask: Grid | null) {
      uniforms.uCT.value.dispose();
      uniforms.uMask.value.dispose();
      uniforms.uCT.value = ct ? gridTexture(ct, false) : emptyTexture();
      uniforms.uMask.value = mask ? gridTexture(mask, true) : emptyTexture();
      uniforms.uHasCT.value = ct ? 1 : 0;
      uniforms.uHasMask.value = mask ? 1 : 0;
      const ext = (ct ?? mask)?.extentsMm ?? [1, 1, 1];
      const m = Math.max(ext[0], ext[1], ext[2]);
      mesh.scale.set(ext[0] / m, ext[2] / m, ext[1] / m);
      dirty = true;
    },
    frame() {
      const moved = controls.update();
      if (moved || dirty) {
        renderer.render(scene, camera);
        dirty = false;
      }
    },
    dispose() {
      controls.dispose();
      uniforms.uCT.value.dispose();
      uniforms.uMask.value.dispose();
      geometry.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Two linked panels                                                           */
/* -------------------------------------------------------------------------- */
type Props = {
  ct: Grid | null;
  mask: Grid | null;
  isDummy?: boolean;
  warning?: string | null;
};

const VIEWS: { name: string; label: string; pos: [number, number, number] }[] = [
  { name: "front", label: "Front", pos: [0, 0, 2.3] },
  { name: "side", label: "Side", pos: [2.3, 0, 0] },
  { name: "top", label: "Top", pos: [0, 2.3, 0.001] },
  { name: "back", label: "Back", pos: [0, 0, -2.3] },
];

export default function Viewer3D({ ct, mask, isDummy = false, warning = null }: Props) {
  const ctHost = useRef<HTMLDivElement>(null);
  const segHost = useRef<HTMLDivElement>(null);
  const rootRef = useRef<HTMLElement>(null);
  const panels = useRef<{ ct: Panel; seg: Panel } | null>(null);
  const [supported, setSupported] = useState(true);
  const [rotating, setRotating] = useState(false);
  const [show, setShow] = useState<[boolean, boolean, boolean]>([true, true, true]);
  const [clip, setClip] = useState(0);
  const [density, setDensity] = useState(1);

  // Create both panels once.
  useEffect(() => {
    if (!ctHost.current || !segHost.current) return;
    const gl = document.createElement("canvas").getContext("webgl2");
    if (!gl) {
      setSupported(false);
      return;
    }
    const steps = window.innerWidth < 768 ? 160 : 256;
    const a = makePanel(ctHost.current, 0, steps);
    const b = makePanel(segHost.current, 1, steps);
    panels.current = { ct: a, seg: b };

    // Keep both cameras identical so the two views can be compared directly.
    let syncing = false;
    const link = (from: Panel, to: Panel) =>
      from.controls.addEventListener("change", () => {
        if (syncing) return;
        syncing = true;
        to.camera.position.copy(from.camera.position);
        to.controls.update();
        to.markDirty();
        syncing = false;
      });
    link(a, b);
    link(b, a);

    // Touching either view stops auto-rotate so it never fights the user.
    const stop = () => {
      a.controls.autoRotate = false;
      setRotating(false);
    };
    a.controls.addEventListener("start", stop);
    b.controls.addEventListener("start", stop);

    const ro = new ResizeObserver(() => {
      a.resize();
      b.resize();
    });
    ro.observe(ctHost.current);
    ro.observe(segHost.current);
    a.resize();
    b.resize();

    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting), { threshold: 0 });
    if (rootRef.current) io.observe(rootRef.current);

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!visible) return;
      a.frame();
      b.frame();
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      a.dispose();
      b.dispose();
      panels.current = null;
    };
  }, []);

  // Upload volumes when they change.
  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    p.ct.setGrids(ct, null);
    p.seg.setGrids(ct, mask);
  }, [ct, mask]);

  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    p.seg.uniforms.uShow.value.set(show[0] ? 1 : 0, show[1] ? 1 : 0, show[2] ? 1 : 0);
    p.seg.markDirty();
  }, [show]);

  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    for (const panel of [p.ct, p.seg]) {
      panel.uniforms.uClip.value = clip;
      panel.uniforms.uDensity.value = density;
      panel.markDirty();
    }
  }, [clip, density]);

  const toggleRotate = useCallback(() => {
    const p = panels.current;
    if (!p) return;
    p.ct.controls.autoRotate = !p.ct.controls.autoRotate;
    setRotating(p.ct.controls.autoRotate);
  }, []);

  const goTo = useCallback((pos: THREE.Vector3 | [number, number, number]) => {
    const p = panels.current;
    if (!p) return;
    p.ct.controls.autoRotate = false;
    setRotating(false);
    if (Array.isArray(pos)) p.ct.camera.position.set(...pos);
    else p.ct.camera.position.copy(pos);
    p.ct.controls.update();
    p.ct.markDirty();
  }, []);

  const toggleLabel = (i: number) =>
    setShow((s) => s.map((v, k) => (k === i ? !v : v)) as [boolean, boolean, boolean]);

  if (!supported) {
    return (
      <div className="ks-card ks-empty" role="alert">
        <h3>3D view is not available</h3>
        <p>This browser does not support WebGL 2. Try a recent version of Chrome, Edge, Firefox or Safari.</p>
      </div>
    );
  }

  return (
    <section ref={rootRef} className="ks-viewer" aria-label="3D viewer">
      <div className="ks-toolbar" role="toolbar" aria-label="Viewer controls">
        <button type="button" className="ks-chip" aria-pressed={rotating} onClick={toggleRotate}>
          {rotating ? "Pause rotation" : "Rotate 360°"}
        </button>
        <div className="ks-seg" role="group" aria-label="View angle">
          {VIEWS.map((v) => (
            <button key={v.name} type="button" className="ks-chip" onClick={() => goTo(v.pos)}>
              {v.label}
            </button>
          ))}
          <button type="button" className="ks-chip" onClick={() => goTo(HOME)}>
            Reset
          </button>
        </div>
        <label className="ks-slider">
          <span>Cut away front</span>
          <input type="range" min={0} max={0.8} step={0.01} value={clip} onChange={(e) => setClip(+e.target.value)} />
        </label>
        <label className="ks-slider">
          <span>Brightness</span>
          <input type="range" min={0.3} max={2.5} step={0.05} value={density} onChange={(e) => setDensity(+e.target.value)} />
        </label>
      </div>

      <div className="ks-panels">
        <figure className="ks-panel">
          <figcaption>
            <strong>Original CT</strong>
            <span>Soft-tissue window, volume rendered</span>
          </figcaption>
          <div className="ks-stage" ref={ctHost}>
            {!ct && <Overlay title="No scan loaded" text="Upload a CT scan or load the dummy demo." />}
            {isDummy && ct && <span className="ks-dummy">Dummy / test data</span>}
            <span className="ks-hint">Drag to rotate. Pinch or scroll to zoom.</span>
          </div>
        </figure>

        <figure className="ks-panel">
          <figcaption>
            <strong>Segmentation overlay</strong>
            <span>Predicted labels on the same scan</span>
          </figcaption>
          <div className="ks-stage" ref={segHost}>
            {ct && !mask && <Overlay title="No segmentation yet" text="Run the model to see the predicted kidney, tumor and cyst." />}
            {!ct && <Overlay title="Waiting for a scan" text="The overlay appears here after segmentation." />}
            {isDummy && mask && <span className="ks-dummy">Dummy / test data</span>}
          </div>
          <div className="ks-legend" role="group" aria-label="Show or hide labels">
            {LABELS.map((l, i) => (
              <button key={l.key} type="button" className="ks-chip ks-legend-chip" aria-pressed={show[i]} onClick={() => toggleLabel(i)}>
                <i style={{ background: l.color }} aria-hidden />
                {l.name}
              </button>
            ))}
          </div>
        </figure>
      </div>

      {warning && (
        <p className="ks-note ks-note-warn" role="status">
          {warning}
        </p>
      )}
    </section>
  );
}

function Overlay({ title, text }: { title: string; text: string }) {
  return (
    <div className="ks-overlay">
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
