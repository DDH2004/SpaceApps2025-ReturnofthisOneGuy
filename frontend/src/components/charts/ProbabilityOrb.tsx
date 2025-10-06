"use client";
import React from "react";

export function ProbabilityOrb({ 
  p, 
  prediction, 
  confidenceMessage 
}: { 
  p: number; 
  prediction?: number;
  confidenceMessage?: string;
}) {
  const v = Math.max(0, Math.min(1, p));
  const pct = (v * 100).toFixed(0);
  
  // Color logic based on prediction and confidence
  let color = "#fbbf24"; // default yellow for uncertain
  if (prediction === 1) { // Predicted as planet
    // Green for high confidence planet, yellow for medium, red for low
    color = v >= 0.8 ? "#34d399" : v >= 0.6 ? "#fbbf24" : "#f87171";
  } else if (prediction === 0) { // Predicted as non-planet
    // For non-planet predictions, green means high confidence it's NOT a planet (low probability)
    // So we invert the logic: low probability = high confidence = green
    color = v <= 0.2 ? "#34d399" : v <= 0.4 ? "#fbbf24" : "#f87171";
  }
  
  // Default message if not provided
  const message = confidenceMessage || (
    prediction === 1 
      ? (v >= 0.8 ? "Very likely planet" : v >= 0.6 ? "Likely planet" : "Possibly planet")
      : prediction === 0
      ? (v <= 0.2 ? "Very likely not a planet" : v <= 0.4 ? "Likely not a planet" : "Possibly not a planet")
      : "Uncertain"
  );
  
  return (
    <div className="flex items-center gap-5">
      <svg
        width="140"
        height="140"
        viewBox="0 0 140 140"
        className="drop-shadow-[0_0_20px_rgba(255,255,255,0.25)]"
      >
        <defs>
          <radialGradient id="g" cx="50%" cy="40%">
            <stop offset="0%" stopColor="white" stopOpacity="0.25" />
            <stop offset="100%" stopColor={color} stopOpacity="0.85" />
          </radialGradient>
        </defs>
        <circle cx="70" cy="70" r="52" fill="url(#g)" opacity="0.25" />
        <circle
          cx="70"
          cy="70"
          r="52"
          stroke="rgba(255,255,255,.15)"
          strokeWidth="10"
          fill="none"
        />
        <circle
          cx="70"
          cy="70"
          r="52"
          stroke={color}
          strokeWidth="10"
          fill="none"
          strokeDasharray={`${2 * Math.PI * 52} ${2 * Math.PI * 52}`}
          strokeDashoffset={`${(1 - v) * 2 * Math.PI * 52}`}
          transform="rotate(-90 70 70)"
        />
        <text
          x="70"
          y="76"
          textAnchor="middle"
          fontSize="24"
          fill="white"
          fontWeight={700}
        >
          {pct}%
        </text>
      </svg>
      <div className="text-sm text-white/80">
        <div className="text-white font-semibold">Model confidence</div>
        <div>{message}</div>
      </div>
    </div>
  );
}
