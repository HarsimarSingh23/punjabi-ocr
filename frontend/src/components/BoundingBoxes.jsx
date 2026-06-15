import { forwardRef } from "react";

// SVG overlay sized to the OCR coordinate space. Each word becomes a polygon
// whose stroke is animated (drawn) by the Workspace orchestrator.
const BoundingBoxes = forwardRef(function BoundingBoxes({ ocr }, ref) {
  if (!ocr) return <svg ref={ref} className="box-layer" />;
  const vw = ocr.width || 1000;
  const vh = ocr.height || 1000;
  return (
    <svg
      ref={ref}
      className="box-layer"
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {ocr.words.map((word, i) =>
        word.box && word.box.length >= 3 ? (
          <polygon
            key={i}
            data-i={i}
            className="word-box"
            points={word.box.map((p) => p.join(",")).join(" ")}
          />
        ) : null
      )}
    </svg>
  );
});

export default BoundingBoxes;
