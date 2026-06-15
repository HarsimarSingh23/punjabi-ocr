import { useState } from "react";
import { AnimatePresence } from "framer-motion";

import AuroraBackground from "./components/AuroraBackground.jsx";
import TopBar from "./components/TopBar.jsx";
import UploadPane from "./components/UploadPane.jsx";
import Workspace from "./components/Workspace.jsx";
import { ToastProvider } from "./lib/useToast.jsx";

export default function App() {
  // null until an image is uploaded, then { id, url }
  const [image, setImage] = useState(null);

  return (
    <ToastProvider>
      <AuroraBackground />
      <div className="app-shell">
        <TopBar />
        <main className="stage">
          <AnimatePresence mode="wait">
            {image ? (
              <Workspace
                key="workspace"
                image={image}
                onReset={() => setImage(null)}
              />
            ) : (
              <UploadPane key="upload" onUploaded={setImage} />
            )}
          </AnimatePresence>
        </main>
      </div>
    </ToastProvider>
  );
}
