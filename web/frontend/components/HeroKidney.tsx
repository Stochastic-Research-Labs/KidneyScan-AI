"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/** Procedural bean-shaped kidney. Pure illustration, not data. */
function kidneyGeometry() {
  const g = new THREE.SphereGeometry(1, 96, 64);
  const pos = g.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    let X = x * 0.55;
    const Y = y;
    X += 0.28 * Y * Y; // ends curve toward the middle
    X -= 0.45 * Math.exp(-(Y * Y) / 0.05) * Math.max(0, x); // hilum notch
    pos.setXYZ(i, X, Y, z * 0.42);
  }
  g.computeVertexNormals();
  return g;
}

export default function HeroKidney() {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.domElement.style.cssText = "display:block;width:100%;height:100%;touch-action:pan-y;";
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 50);
    camera.position.set(0, 0.2, 6.4);
    scene.add(new THREE.AmbientLight(0xaecbd6, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(3, 4, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x4fd1c5, 2.2);
    rim.position.set(-4, 1, -3);
    scene.add(rim);

    const geo = kidneyGeometry();
    const tissue = new THREE.MeshPhysicalMaterial({
      color: 0x3b7f8a, roughness: 0.42, metalness: 0.05, clearcoat: 0.7, clearcoatRoughness: 0.3,
    });
    const group = new THREE.Group();
    const right = new THREE.Mesh(geo, tissue);
    right.position.set(-1.05, 0, 0);
    right.rotation.z = -0.18;
    const left = new THREE.Mesh(geo, tissue);
    left.position.set(1.05, 0, 0);
    left.scale.x = -1;
    left.rotation.z = 0.18;
    const tumor = new THREE.Mesh(
      new THREE.SphereGeometry(0.21, 32, 24),
      new THREE.MeshStandardMaterial({ color: 0xff5d5d, emissive: 0x5a1010, roughness: 0.5 }),
    );
    tumor.position.set(1.52, 0.3, 0.05);
    const cyst = new THREE.Mesh(
      new THREE.SphereGeometry(0.14, 32, 24),
      new THREE.MeshPhysicalMaterial({ color: 0xffc83d, roughness: 0.2, transparent: true, opacity: 0.9, clearcoat: 1 }),
    );
    cyst.position.set(-1.5, -0.35, 0.1);
    group.add(right, left, tumor, cyst);
    group.rotation.x = 0.1;
    scene.add(group);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.enableDamping = true;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    controls.autoRotate = !reduce;
    controls.autoRotateSpeed = 1.2;
    renderer.domElement.style.touchAction = "pan-y";

    const resize = () => {
      const w = el.clientWidth || 1, h = el.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();

    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(el);
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      if (!visible) return;
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      controls.dispose();
      geo.dispose();
      tissue.dispose();
      tumor.geometry.dispose();
      cyst.geometry.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, []);

  return <div ref={host} className="ks-hero-canvas" role="img" aria-label="Rotating illustration of two kidneys, one with a tumor and one with a cyst" />;
}
