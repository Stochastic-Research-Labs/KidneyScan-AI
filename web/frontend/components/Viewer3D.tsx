"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { Grid } from "../lib/nifti";
import { LABELS } from "../lib/labels";

/* -------------------------------------------------------------------------- */
/* Slice-stack volume viewer.                                                  */
/* The old custom GLSL raymarcher rendered blank in every browser, even with   */
/* synthetic data and the exact same shader, so it is replaced by a stack of   */
/* textured planes (one per axial slice). It uses only basic three.js          */
/* materials, so there is no shader that can silently fail.                    */
/* Display axes: 0 = Left (x), 1 = Anterior (z), 2 = Superior (y).            */
/* Grid data index = ((c * sy + b) * sx + a), a = Left, b = Anterior,          */
/* c = Superior. Each plane is a Left x Anterior cross-section, stacked        */
/* along Superior.                                                             */
/* -------------------------------------------------------------------------- */

const MAX_PLANES = 40;
const HOME = new THREE.Vector3(0.9, 0.45, 2.3);

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

const hexToRgb = (h: string): [number, number, number] => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];

const LABEL_RGB = LABELS.map((l) => hexToRgb(l.color));

/** Evenly spaced Superior indices, at most MAX_PLANES of them. */
function sliceIndices(sz: number): number[] {
  const n = Math.min(MAX_PLANES, sz);
  if (n <= 1) return [0];
  const out: number[] = [];
  for (let i = 0; i < n; i++) out.push(Math.round((i * (sz - 1)) / (n - 1)));
  return out;
}

function makeTexture(rgba: Uint8Array<ArrayBuffer>, w: number, h: number): THREE.DataTexture {
  const t = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.minFilter = THREE.LinearFilter;
  t.magFilter = THREE.LinearFilter;
  t.wrapS = THREE.ClampToEdgeWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** CT slice: grey value, alpha = value (air transparent, tissue visible). */
function ctSliceRGBA(g: Grid, c: number): { data: Uint8Array<ArrayBuffer>; w: number; h: number } {
  const [sx, sy] = g.size;
  const data = new Uint8Array(sx * sy * 4);
  const base = c * sx * sy;
  for (let b = 0; b < sy; b++) {
    for (let a = 0; a < sx; a++) {
      const v = g.data[base + b * sx + a];
      const i = (b * sx + a) * 4;
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
      data[i + 3] = v;
    }
  }
  return { data, w: sx, h: sy };
}

/** Overlay slice: dimmed CT underneath, label colour on top where shown. */
function overlaySliceRGBA(
  ct: Grid,
  mask: Grid,
  c: number,
  show: [boolean, boolean, boolean],
): { data: Uint8Array<ArrayBuffer>; w: number; h: number } {
  const [sx, sy] = ct.size;
  const data = new Uint8Array(sx * sy * 4);
  const base = c * sx * sy;
  for (let b = 0; b < sy; b++) {
    for (let a = 0; a < sx; a++) {
      const i = (b * sx + a) * 4;
      const lab = mask.data[base + b * sx + a];
      if (lab >= 1 && lab <= 3 && show[lab - 1]) {
        const [r, gg, bb] = LABEL_RGB[lab - 1];
        data[i] = r;
        data[i + 1] = gg;
        data[i + 2] = bb;
        data[i + 3] = 235;
      } else {
        const v = Math.round(ct.data[base + b * sx + a] * 0.55);
        data[i] = v;
        data[i + 1] = v;
        data[i + 2] = v;
        data[i + 3] = v;
      }
    }
  }
  return { data, w: sx, h: sy };
}

/* -------------------------------------------------------------------------- */
/* One WebGL panel                                                             */
/* -------------------------------------------------------------------------- */

type Panel = {
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  group: THREE.Group;
  planes: THREE.Mesh[];
  materials: THREE.MeshBasicMaterial[];
  geometry: THREE.BufferGeometry | null;
  setStacks: (
    ct: Grid | null,
    mask: Grid | null,
    show: [boolean, boolean, boolean],
    brightness: number,
    clip: number,
  ) => void;
  setBrightness: (b: number) => void;
  setClip: (c: number) => void;
  resize: () => void;
  dispose: () => void;
};

function makePanel(host: HTMLElement, mode: 0 | 1): Panel {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  const canvas = renderer.domElement;
  canvas.style.cssText = "display:block;width:100%;height:100%;touch-action:none;outline:none;";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    mode === 0
      ? "3D view of the original CT. Drag to rotate."
      : "3D view of the segmentation overlay. Drag to rotate.",
  );
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

  const group = new THREE.Group();
  scene.add(group);

  const panel: Panel = {
    camera,
    controls,
    group,
    planes: [],
    materials: [],
    geometry: null,

    setStacks(ct, mask, show, brightness, clip) {
      // Clear the old stack.
      for (const p of panel.planes) {
        const m = p.material as THREE.MeshBasicMaterial;
        m.map?.dispose();
        m.dispose();
        group.remove(p);
      }
      panel.planes = [];
      panel.materials = [];
      panel.geometry?.dispose();
      panel.geometry = null;

      const grid = mode === 0 ? ct : ct;
      if (!grid) return;
      if (mode === 1 && !mask) return;

      const [sx, sy, sz] = grid.size;
      const m = Math.max(grid.extentsMm[0], grid.extentsMm[1], grid.extentsMm[2]);
      const w = grid.extentsMm[0] / m; // Left
      const h = grid.extentsMm[1] / m; // Anterior
      const depth = grid.extentsMm[2] / m; // Superior
      const geo = new THREE.PlaneGeometry(w, h);
      geo.rotateX(-Math.PI / 2); // lie flat: Left x Anterior, stacked along Superior
      panel.geometry = geo;

      const idx = sliceIndices(sz);
      const keep = Math.max(1, Math.round(idx.length * (1 - clip)));
      idx.forEach((c, k) => {
        const slice =
          mode === 0
            ? ctSliceRGBA(grid, c)
            : overlaySliceRGBA(grid, mask as Grid, c, show);
        const tex = makeTexture(slice.data, slice.w, slice.h);
        const mat = new THREE.MeshBasicMaterial({
          map: tex,
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        mat.color.setScalar(Math.min(Math.max(brightness, 0.1), 3));
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.y = (c / Math.max(sz - 1, 1) - 0.5) * depth;
        mesh.visible = k < keep;
        group.add(mesh);
        panel.planes.push(mesh);
        panel.materials.push(mat);
      });
    },

    setBrightness(b: number) {
      const v = Math.min(Math.max(b, 0.1), 3);
      for (const mat of panel.materials) mat.color.setScalar(v);
    },

    setClip(c: number) {
      const keep = Math.max(1, Math.round(panel.planes.length * (1 - c)));
      panel.planes.forEach((p, k) => (p.visible = k < keep));
    },

    resize() {
      const w = host.clientWidth || 1;
      const h = host.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    },

    dispose() {
      controls.dispose();
      for (const p of panel.planes) {
        const mat = p.material as THREE.MeshBasicMaterial;
        mat.map?.dispose();
        mat.dispose();
      }
      panel.geometry?.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      canvas.remove();
    },
  };

  // Render loop is driven by the component's tick; keep a handle to render.
  (panel as unknown as { render: () => void }).render = () => renderer.render(scene, camera);
  return panel;
}

/* -------------------------------------------------------------------------- */
/* Two linked panels                                                           */
/* -------------------------------------------------------------------------- */

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
    let a: Panel;
    let b: Panel;
    try {
      a = makePanel(ctHost.current, 0);
      b = makePanel(segHost.current, 1);
    } catch {
      setSupported(false);
      return;
    }
    panels.current = { ct: a, seg: b };

    // Keep both cameras identical so the two views can be compared directly.
    let syncing = false;
    const link = (from: Panel, to: Panel) =>
      from.controls.addEventListener("change", () => {
        if (syncing) return;
        syncing = true;
        to.camera.position.copy(from.camera.position);
        to.controls.update();
        syncing = false;
      });
    link(a, b);
    link(b, a);

    // Touching either view stops auto-rotate so it never fights the user.
    const stop = () => {
      a.controls.autoRotate = false;
      b.controls.autoRotate = false;
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
      a.controls.update();
      b.controls.update();
      (a as unknown as { render: () => void }).render();
      (b as unknown as { render: () => void }).render();
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

  // Rebuild the stacks when the volumes or the label toggles change.
  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    p.ct.setStacks(ct, mask, show, density, clip);
    p.seg.setStacks(ct, mask, show, density, clip);
  }, [ct, mask, show]);

  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    p.ct.setBrightness(density);
    p.seg.setBrightness(density);
  }, [density]);

  useEffect(() => {
    const p = panels.current;
    if (!p) return;
    p.ct.setClip(clip);
    p.seg.setClip(clip);
  }, [clip]);

  const toggleRotate = useCallback(() => {
    const p = panels.current;
    if (!p) return;
    const next = !p.ct.controls.autoRotate;
    p.ct.controls.autoRotate = next;
    p.seg.controls.autoRotate = next;
    setRotating(next);
  }, []);

  const goTo = useCallback((pos: THREE.Vector3 | [number, number, number]) => {
    const p = panels.current;
    if (!p) return;
    p.ct.controls.autoRotate = false;
    p.seg.controls.autoRotate = false;
    setRotating(false);
    if (Array.isArray(pos)) p.ct.camera.position.set(...pos);
    else p.ct.camera.position.copy(pos);
    p.ct.controls.update();
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
