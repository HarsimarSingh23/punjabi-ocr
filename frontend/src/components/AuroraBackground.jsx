import { motion } from "framer-motion";

// Slow-drifting colored blobs behind the whole app — the ambient "effect".
const BLOBS = [
  { className: "blob amber", x: [-40, 60, -40], y: [-30, 40, -30], d: 26 },
  { className: "blob teal", x: [40, -50, 40], y: [20, -40, 20], d: 32 },
  { className: "blob violet", x: [-20, 30, -20], y: [50, -20, 50], d: 38 },
];

export default function AuroraBackground() {
  return (
    <div className="aurora" aria-hidden="true">
      {BLOBS.map((b, i) => (
        <motion.div
          key={i}
          className={b.className}
          animate={{ x: b.x, y: b.y }}
          transition={{ duration: b.d, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
      <div className="aurora-grain" />
    </div>
  );
}
