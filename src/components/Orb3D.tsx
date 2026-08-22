import React, { useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbFallback } from './OrbFallback';
import * as THREE from 'three';

interface Orb3DProps {
  state: 'IDLE' | 'LISTENING' | 'PROCESSING' | 'ANSWERING' | 'ERROR';
  audioAmplitude?: number;
}

const AnimatedSphere: React.FC<Orb3DProps> = ({ state, audioAmplitude = 0 }) => {
  const meshRef = useRef<THREE.Mesh>(null!);
  const outerWireRef = useRef<THREE.Mesh>(null!);

  useFrame(({ clock }) => {
    const time = clock.getElapsedTime();
    if (meshRef.current) {
      meshRef.current.rotation.y = time * 0.4;
      meshRef.current.rotation.x = Math.sin(time * 0.2) * 0.2;

      let targetScale = 1.6;
      if (state === 'LISTENING') {
        targetScale = 1.6 + audioAmplitude * 0.8 + Math.sin(time * 6) * 0.1;
      } else if (state === 'PROCESSING') {
        targetScale = 1.7 + Math.sin(time * 10) * 0.15;
      } else if (state === 'ANSWERING') {
        targetScale = 1.75 + Math.sin(time * 4) * 0.08;
      } else if (state === 'ERROR') {
        targetScale = 1.4;
      }

      meshRef.current.scale.set(targetScale, targetScale, targetScale);
    }

    if (outerWireRef.current) {
      outerWireRef.current.rotation.z = time * 0.5;
      outerWireRef.current.rotation.y = -time * 0.3;
    }
  });

  let mainColor = "#38bdf8"; // Cyan
  if (state === "LISTENING") mainColor = "#00f3ff";
  else if (state === "PROCESSING") mainColor = "#a855f7"; // Purple
  else if (state === "ANSWERING") mainColor = "#10b981"; // Emerald
  else if (state === "ERROR") mainColor = "#ef4444"; // Red

  return (
    <group>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1.5} color={mainColor} />
      
      {/* Core Shader-like Mesh */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color={mainColor}
          roughness={0.15}
          metalness={0.8}
          wireframe={state === "PROCESSING"}
          emissive={mainColor}
          emissiveIntensity={state === "LISTENING" ? 0.8 : 0.4}
        />
      </mesh>

      {/* Futuristic Orbiting Outer Ring */}
      <mesh ref={outerWireRef}>
        <torusGeometry args={[2.2, 0.03, 16, 100]} />
        <meshBasicMaterial color={mainColor} wireframe />
      </mesh>
    </group>
  );
};

export const Orb3D: React.FC<Orb3DProps> = ({ state, audioAmplitude = 0 }) => {
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return <OrbFallback state={state} audioAmplitude={audioAmplitude} />;
  }

  return (
    <div className="relative w-72 h-72 mx-auto flex items-center justify-center">
      <Canvas
        className="w-full h-full"
        camera={{ position: [0, 0, 5], fov: 60 }}
        onError={() => setHasError(true)}
      >
        <AnimatedSphere state={state} audioAmplitude={audioAmplitude} />
      </Canvas>
    </div>
  );
};
