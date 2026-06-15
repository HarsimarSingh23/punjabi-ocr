import { Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";

export default function TopBar() {
  const onAdmin = useLocation().pathname === "/admin";
  return (
    <motion.header
      className="topbar"
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Link to="/" className="brand">
        <span className="logo">ੴ</span>
        <span className="brand-name">
          Punjabi OCR
          <span className="brand-sub" lang="pa">
            {onAdmin ? "Admin" : "ਪੰਜਾਬੀ ਓ.ਸੀ.ਆਰ."}
          </span>
        </span>
      </Link>
      {onAdmin ? (
        <Link className="admin-link" to="/">
          ← Back to app
        </Link>
      ) : (
        <Link className="admin-link" to="/admin">
          ⚙ Admin
        </Link>
      )}
    </motion.header>
  );
}
