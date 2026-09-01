import os
import re
from typing import Any, Dict, List, Optional

import pymupdf as fitz
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
)


class FigureExtractor:
    """
    Vision-first figure extractor.

    Uses the Aryn/deformable-detr-DocLayNet layout model to detect
    'Picture' regions on each page (captures raster images, vector
    charts, flowcharts, and diagrams alike).

    pdfplumber/pymupdf are used only to:
      1. Render each page to a PIL image for model inference.
      2. Crop and save the detected region at high resolution.
      3. Extract text blocks for caption matching.
    """

    MODEL_NAME = "Aryn/deformable-detr-DocLayNet"

    # DocLayNet label for figures / pictures / diagrams.
    PICTURE_LABEL = "Picture"

    # Confidence threshold for detection.
    # DocLayNet is well-calibrated; 0.60 keeps flowcharts/diagrams
    # while avoiding stray decorations.
    DEFAULT_CONFIDENCE = 0.60

    # Scale factor for page → PIL image (for detection).
    DETECTION_SCALE = 2

    # Scale factor for the final saved crop (higher = sharper).
    SAVE_SCALE = 3

    # Minimum region size in PDF points to bother saving.
    MIN_WIDTH = 40.0
    MIN_HEIGHT = 40.0

    # Fraction of near-white pixels that makes an image "blank".
    BLANK_THRESHOLD = 0.97

    def __init__(
        self,
        doc,
        output_dir: str = "output/figures",
        model_name: str = MODEL_NAME,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        device: Optional[str] = None,
    ):
        self.doc = doc
        self.output_dir = output_dir
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        print(f"[FigureExtractor] Device: {self.device}")

        # ---------------------------------------------------------
        # Load layout model
        # ---------------------------------------------------------

        self._load_model()

    def _load_model(self) -> None:
        print(f"[FigureExtractor] Loading model: {self.model_name}")
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_name
        )
        self.model.to(self.device)
        self.model.eval()

        # Resolve the numeric ID for the 'Picture' class.
        self._picture_id = None
        for label_id, label_name in self.model.config.id2label.items():
            if label_name == self.PICTURE_LABEL:
                self._picture_id = int(label_id)
                break

        if self._picture_id is None:
            raise RuntimeError(
                f"Label '{self.PICTURE_LABEL}' not found in model. "
                f"Available: {self.model.config.id2label}"
            )

        print(
            f"[FigureExtractor] '{self.PICTURE_LABEL}' "
            f"label id: {self._picture_id}"
        )
        print("[FigureExtractor] Model loaded.")

    # =========================================================
    # PAGE RENDERING
    # =========================================================

    def _render_page(self, page, scale: int) -> Image.Image:
        """Render a PyMuPDF page to a PIL RGB image."""
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    def _render_region(
        self,
        page,
        pdf_bbox: List[float],
    ) -> Image.Image:
        """Render a specific region of a page at high resolution."""
        x0, y0, x1, y1 = pdf_bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        mat = fitz.Matrix(self.SAVE_SCALE, self.SAVE_SCALE)
        pix = page.get_pixmap(clip=rect, matrix=mat)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # =========================================================
    # VISION DETECTION
    # =========================================================

    def _detect_figures(
        self,
        image: Image.Image,
    ) -> List[Dict[str, Any]]:
        """
        Run the layout model and return bounding boxes (in pixel
        coordinates of 'image') for all detected Picture regions.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # post_process_object_detection returns boxes in [x0, y0, x1, y1]
        # pixel format for the given target_sizes.
        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            if int(label.item()) == self._picture_id:
                detections.append(
                    {
                        "bbox": [round(v, 2) for v in box.tolist()],
                        "score": round(float(score.item()), 4),
                    }
                )

        return detections

    # =========================================================
    # COORDINATE CONVERSION
    # =========================================================

    def _image_bbox_to_pdf(
        self,
        image_bbox: List[float],
        image: Image.Image,
        page,
    ) -> List[float]:
        """
        Convert a bounding box in image-pixel coordinates to PDF points.
        """
        img_w, img_h = image.size
        pdf_w = page.rect.width
        pdf_h = page.rect.height

        x0, y0, x1, y1 = image_bbox
        return [
            x0 / img_w * pdf_w,
            y0 / img_h * pdf_h,
            x1 / img_w * pdf_w,
            y1 / img_h * pdf_h,
        ]

    # =========================================================
    # CONTENT GATE
    # =========================================================

    @staticmethod
    def _is_blank(img: Image.Image, threshold: float) -> bool:
        """
        Return True if the image is mostly white / near-white.

        Uses pure PIL — no numpy dependency.
        """
        gray = img.convert("L")
        pixels = gray.getdata()
        white = sum(1 for p in pixels if p >= 240)
        return (white / len(pixels)) >= threshold

    @staticmethod
    def _is_too_small(pdf_bbox: List[float], min_w: float, min_h: float) -> bool:
        x0, y0, x1, y1 = pdf_bbox
        return (x1 - x0) < min_w or (y1 - y0) < min_h

    # =========================================================
    # CAPTION DETECTION
    # =========================================================

    @staticmethod
    def is_figure_caption(text: str) -> bool:
        return bool(
            re.match(
                r"^\s*(figure|fig\.?)\s*\d+",
                text,
                re.IGNORECASE,
            )
        )

    # Inline body-text references look like "Fig. 8 presents..."
    # or "Figure 12 shows...".  These are NOT captions.
    BODY_REF_RE = re.compile(
        r"^\s*(figure|fig\.?)\s*\d+\s+"
        r"(presents|shows|illustrates|depicts|gives|demonstrates"
        r"|compares|displays|plots|contains|reports|summarizes)",
        re.IGNORECASE,
    )

    def find_caption(
        self,
        page,
        image_bbox: List[float],
        max_caption_distance: float = 150.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a 'Figure N ...' caption in two passes:

        Pass 1 – below the detected region (preferred): looks for a
                 caption text block whose top is within
                 max_caption_distance points below the figure bottom.

        Pass 2 – inside the detected region: DocLayNet's 'Picture'
                 bounding box often includes the caption itself.
                 If pass 1 found nothing, search text blocks whose
                 top falls inside the figure bbox.

        BODY_REF_RE filters inline references like 'Fig. 8 presents...'
        """
        text_blocks = page.get_text("blocks")
        image_x0, image_y0, image_x1, image_y1 = image_bbox

        below: list = []
        inside: list = []

        for block in text_blocks:
            text = block[4].replace("\n", " ").strip()
            if not text:
                continue
            if not self.is_figure_caption(text):
                continue
            if self.BODY_REF_RE.match(text):
                continue

            bx0, by0, bx1, by1 = block[:4]
            horizontal_overlap = max(
                0,
                min(image_x1, bx1) - max(image_x0, bx0),
            )
            if horizontal_overlap <= 0:
                continue

            if by0 >= image_y1:
                # Pass 1: caption is below the figure
                dist = by0 - image_y1
                if dist <= max_caption_distance:
                    below.append((dist, text, [bx0, by0, bx1, by1]))

            elif by0 >= image_y0:
                # Pass 2: caption is inside the detected region
                # (distance from the bottom of the detected box)
                dist = image_y1 - by0
                inside.append((dist, text, [bx0, by0, bx1, by1]))

        # Prefer captions found below; fall back to inside.
        candidates = below or inside

        if candidates:
            candidates.sort(key=lambda c: c[0])
            return {
                "caption": candidates[0][1],
                "bbox": candidates[0][2],
            }

        return None

    # =========================================================
    # MAIN EXTRACTION
    # =========================================================

    def extract_figures(self) -> List[Dict[str, Any]]:
        """
        Extract all vision-detected figures from the document.

        Captures raster images, vector drawings, flowcharts,
        plots, and diagrams — anything the layout model labels
        as 'Picture'.
        """
        figures = []

        for page_num, page in enumerate(self.doc):
            print(
                f"[FigureExtractor] "
                f"Processing page {page_num + 1}/{len(self.doc)}"
            )

            # 1. Render page for detection
            try:
                page_image = self._render_page(
                    page,
                    scale=self.DETECTION_SCALE,
                )
            except Exception as exc:
                print(
                    f"[WARNING] Could not render page "
                    f"{page_num + 1}: {exc}"
                )
                continue

            # 2. Detect figure regions
            detections = self._detect_figures(page_image)
            print(
                f"[FigureExtractor] "
                f"Page {page_num + 1}: "
                f"{len(detections)} figure(s) detected"
            )

            saved_idx = 0

            for detection in detections:
                score = detection["score"]

                # 3. Convert to PDF coordinates
                pdf_bbox = self._image_bbox_to_pdf(
                    detection["bbox"],
                    page_image,
                    page,
                )

                # 4. Reject regions that are too small
                if self._is_too_small(
                    pdf_bbox, self.MIN_WIDTH, self.MIN_HEIGHT
                ):
                    print(
                        f"[FigureExtractor] "
                        f"Page {page_num + 1}: "
                        f"skipped (too small, conf={score:.2f})"
                    )
                    continue

                # 5. Render the region at high resolution
                try:
                    region_img = self._render_region(page, pdf_bbox)
                except Exception as exc:
                    print(
                        f"[WARNING] Could not render region "
                        f"on page {page_num + 1}: {exc}"
                    )
                    continue

                # 6. Reject blank / white images
                if self._is_blank(region_img, self.BLANK_THRESHOLD):
                    print(
                        f"[FigureExtractor] "
                        f"Page {page_num + 1}: "
                        f"skipped (blank image, conf={score:.2f})"
                    )
                    continue

                # 7. Save as PNG
                fig_id = f"fig_p{page_num + 1}_{saved_idx}"
                img_name = f"{fig_id}.png"
                img_path = os.path.join(self.output_dir, img_name)
                region_img.save(img_path, "PNG")

                # 8. Find caption metadata
                caption_data = self.find_caption(page, pdf_bbox)
                if caption_data:
                    caption = caption_data["caption"]
                    caption_bbox = caption_data["bbox"]
                else:
                    caption = (
                        f"Figure {saved_idx + 1} "
                        f"on Page {page_num + 1}"
                    )
                    caption_bbox = None

                figures.append(
                    {
                        "figure_id": fig_id,
                        "page": page_num + 1,
                        "image_path": img_path,
                        "bbox": pdf_bbox,
                        "caption": caption,
                        "caption_bbox": caption_bbox,
                        "detection_confidence": score,
                    }
                )

                saved_idx += 1

        print(
            f"[FigureExtractor] "
            f"Total figures extracted: {len(figures)}"
        )
        return figures
