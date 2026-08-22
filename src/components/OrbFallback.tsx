import React, { useEffect, useRef } from 'react';

interface OrbProps {
  state: 'IDLE' | 'LISTENING' | 'PROCESSING' | 'ANSWERING' | 'ERROR';
  audioAmplitude?: number;
}

export const OrbFallback: React.FC<OrbProps> = ({ state, audioAmplitude = 0 }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;
    let time = 0;

    const render = () => {
      time += 0.03;
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      ctx.clearRect(0, 0, width, height);

      let baseRadius = 75;
      let primaryColor = '#38bdf8'; // Cyan
      let glowColor = 'rgba(56, 189, 248, 0.4)';

      if (state === 'LISTENING') {
        baseRadius = 75 + audioAmplitude * 35 + Math.sin(time * 8) * 6;
        primaryColor = '#00f3ff';
        glowColor = 'rgba(0, 243, 255, 0.6)';
      } else if (state === 'PROCESSING') {
        baseRadius = 80 + Math.sin(time * 12) * 8;
        primaryColor = '#a855f7'; // Purple
        glowColor = 'rgba(168, 85, 247, 0.5)';
      } else if (state === 'ANSWERING') {
        baseRadius = 82 + Math.sin(time * 5) * 5;
        primaryColor = '#10b981'; // Emerald
        glowColor = 'rgba(16, 185, 129, 0.5)';
      } else if (state === 'ERROR') {
        baseRadius = 70;
        primaryColor = '#ef4444'; // Red
        glowColor = 'rgba(239, 68, 68, 0.5)';
      }

      // Draw Outer Glow Aura
      const gradient = ctx.createRadialGradient(centerX, centerY, baseRadius * 0.4, centerX, centerY, baseRadius * 1.8);
      gradient.addColorStop(0, primaryColor);
      gradient.addColorStop(0.5, glowColor);
      gradient.addColorStop(1, 'rgba(6, 10, 18, 0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius * 1.8, 0, Math.PI * 2);
      ctx.fill();

      // Draw Core Orb Sphere
      const coreGrad = ctx.createRadialGradient(centerX - 20, centerY - 20, 10, centerX, centerY, baseRadius);
      coreGrad.addColorStop(0, '#ffffff');
      coreGrad.addColorStop(0.4, primaryColor);
      coreGrad.addColorStop(1, '#0f172a');

      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
      ctx.fill();

      // Draw Animated Orbiting Particle Rings
      ctx.strokeStyle = primaryColor;
      ctx.lineWidth = 2;
      ctx.save();
      ctx.translate(centerX, centerY);
      ctx.rotate(time * 0.8);
      ctx.beginPath();
      ctx.ellipse(0, 0, baseRadius * 1.35, baseRadius * 0.45, Math.PI / 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [state, audioAmplitude]);

  return (
    <div className="relative w-64 h-64 mx-auto flex items-center justify-center">
      <canvas ref={canvasRef} width={256} height={256} className="w-full h-full" />
    </div>
  );
};
