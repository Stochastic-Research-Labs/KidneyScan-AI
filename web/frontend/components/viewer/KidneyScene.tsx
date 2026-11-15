"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { CLASS_COLORS, type Lesion } from "@/lib/api";

/**
 * KidneyScene — adapted from NeuroScan's BrainScene pattern (R3F Canvas +
 * OrbitControls, lights, one merged organ mesh). Differences, and why:
 * - No .glb file: the kidney is a scaled sphere (bean shape), because no
 *   kidney .glb exists in this repo. The shape is schematic, not anatomy.
 * - Lesions are spheres placed from the prediction's lesions[] list
 *   (centre + radius from the model output). Nothing is hard-coded:
 *   a result with no lesions renders no highlight at all.
 * - Left panel passes showOverlay=false, so it renders kidney only.
 */
function KidneyMesh({ lesions, showOverlay }: { lesions: Lesion[]; showOverlay: boolean }) {
  return (
    <group>
      {/* Kidney, grey. Slightly transparent on the right so lesions inside show. */}
      <mesh scale={[0.9, 1.5, 0.75]}>
        <sphereGeometry args={[1, 48, 48]} />
        <meshStandardMaterial
          color={CLASS_COLORS.kidney}
          roughness={0.55}
          metalness={0.1}
          transparent={showOverlay}
          opacity={showOverlay ? 0.45 : 1}
        />
      </mesh>
      {showOverlay &&
        lesions.map((lesion, i) => {
          const [d, h, w] = lesion.center_xyz;
          const pos: [number, number, number] = [(w - 0.5) * 2.6, (0.5 - h) * 3.6, (d - 0.5) * 2.6];
          const r = Math.max(0.12, lesion.radius_mm / 40);
          const isTumor = lesion.class === "tumor";
          return (
            <group key={`${lesion.class}-${i}`} position={pos}>
              <mesh>
                <sphereGeometry args={[r, 32, 32]} />
                <meshStandardMaterial
                  color={isTumor ? CLASS_COLORS.tumor : CLASS_COLORS.cyst}
                  roughness={0.35}
                  emissive={isTumor ? CLASS_COLORS.tumor : CLASS_COLORS.cyst}
                  emissiveIntensity={0.35}
                />
              </mesh>
              {isTumor && (
                <mesh>
                  <sphereGeometry args={[r * 0.55, 24, 24]} />
                  <meshStandardMaterial color={CLASS_COLORS.tumorCore} roughness={0.3} emissive={CLASS_COLORS.tumorCore} emissiveIntensity={0.5} />
                </mesh>
              )}
            </group>
          );
        })}
    </group>
  );
}

export function KidneyScene({
  lesions,
  showOverlay,
  autoRotate,
  resetKey,
}: {
  lesions: Lesion[];
  showOverlay: boolean;
  autoRotate: boolean;
  resetKey: number;
}) {
  return (
    <div className="h-[380px] w-full sm:h-[440px]">
      <Canvas
        key={resetKey}
        camera={{ position: [0, 0.6, 5.2], fov: 40 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
      >
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 4, 5]} intensity={2.2} color="#d5f6ff" />
        <directionalLight position={[-4, 1, -2]} intensity={1.5} color="#22d3ee" />
        <KidneyMesh lesions={lesions} showOverlay={showOverlay} />
        {/* Drag = rotate 360, scroll = zoom, right-drag = pan. Same controls as NeuroScan. */}
        <OrbitControls makeDefault autoRotate={autoRotate} autoRotateSpeed={1.2} enableDamping dampingFactor={0.1} minDistance={2.5} maxDistance={10} />
      </Canvas>
    </div>
  );
}
