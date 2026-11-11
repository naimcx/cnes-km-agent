---
name: cad-inspector
description: Metrological inspection and technical drawing analysis guide for launch vehicles.
---

# CAD & Technical Drawing Inspection Guidelines

This skill guides the visual analysis of engineering drawings, assembly blueprints, and dimensioned diagrams from the flight manual.

## Associated Commands and Tools
- To visually inspect a page: invoke `inspect_pdf_page_image(page_number, query)`.

## Analysis and Transcription Rules
1. Identify the type of visual: CAD dimensional drawing, exploded hardware view, curve plot, or photo.
2. Extract all visible callouts, labels, legends, and captions verbatim.
3. For dimensional drawings, carefully transcribe all diameter symbols (Ø), radius, and vertical/horizontal dimensions (in mm or m).
4. For hardware schematics, list every annotated part, quantity, and functional name.
5. For plots/graphs, extract axis titles, units, limits, and curve levels.
6. Zero-hallucination guarantee: Only report what is clearly visible on the image. If a dimension, value, or formula is not visibly printed, explicitly state: 'Not shown on the visual'.
