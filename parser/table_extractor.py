"""
Vision-first table extraction.

Pipeline:

    PDF
      |
      v
    Render page to image
      |
      v
    Microsoft Table Transformer (TATR)
      |
      |  "this region looks like a table"
      v
    Table bounding boxes
      |
      v
    pdfplumber crop using the detected PDF coordinates
      |
      v
    Text extraction inside confirmed table region
      |
      v
    Caption attachment as metadata
      |
      v
    JSON-compatible table records

The detector, NOT pdfplumber geometry, decides whether a table exists.

Recommended model:
    microsoft/table-transformer-detection

Install:
    pip install pdfplumber torch torchvision transformers pillow

Optional:
    pip install accelerate

Usage from main.py:

    extractor = TableExtractor(
        pdf_path="data/processed/2012.12104v1/document.pdf",
        output_dir="data/processed/2012.12104v1/tables",
    )

    tables = extractor.extract_tables()

"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
import torch

from PIL import Image

from transformers import (
    AutoImageProcessor,
    TableTransformerForObjectDetection,
)


class TableExtractor:
    """
    Vision-first table extractor.

    The vision model determines table regions.
    pdfplumber is used only to extract text from those regions.
    """

    MODEL_NAME = "microsoft/table-transformer-detection"

    # Detection confidence.
    # 0.65 avoids page footers, figure captions, and other low-confidence
    # detections while still catching real tables at 0.80+.
    DEFAULT_CONFIDENCE = 0.65

    # Extra PDF points around the detected table.
    # This prevents tight detector boxes from cutting off text.
    DEFAULT_PADDING = 4.0

    # Minimum detected table size.
    MIN_TABLE_WIDTH = 50.0
    MIN_TABLE_HEIGHT = 15.0

    # Caption detection is metadata only.
    CAPTION_RE = re.compile(
        r"^\s*table\s+(\d+)\s*[\.:]?\s*(.*)$",
        re.IGNORECASE,
    )

    FIGURE_RE = re.compile(
        r"^\s*(figure|fig\.?)\s+\d+",
        re.IGNORECASE,
    )

    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "output/tables",
        model_name: str = MODEL_NAME,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        padding: float = DEFAULT_PADDING,
        device: Optional[str] = None,
    ):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.padding = padding

        os.makedirs(self.output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        print(
            f"[TableExtractor] Device: {self.device}"
        )

        # ---------------------------------------------------------
        # Load vision model
        # ---------------------------------------------------------

        print(
            f"[TableExtractor] Loading model: "
            f"{self.model_name}"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name
        )

        self.model = (
            TableTransformerForObjectDetection
            .from_pretrained(self.model_name)
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            "[TableExtractor] Vision model loaded."
        )

    # =========================================================
    # TEXT HELPERS
    # =========================================================

    @staticmethod
    def _clean_text(text: Any) -> str:
        if text is None:
            return ""

        text = str(text)

        text = re.sub(
            r"[\r\n\t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    @classmethod
    def _is_caption(cls, text: str) -> bool:
        if not text:
            return False

        return (
            cls.CAPTION_RE.match(
                cls._clean_text(text)
            )
            is not None
        )

    @classmethod
    def _caption_number(
        cls,
        text: str,
    ) -> Optional[int]:

        match = cls.CAPTION_RE.match(
            cls._clean_text(text)
        )

        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None

    # =========================================================
    # PAGE RENDERING
    # =========================================================

    def _render_page(
        self,
        page,
        resolution: int = 150,
    ) -> Image.Image:
        """
        Render a PDF page into a PIL image.

        pdfplumber uses PDF coordinates.
        The rendered image is only used by the vision detector.
        """

        page_image = page.to_image(
            resolution=resolution
        )

        return page_image.original.convert(
            "RGB"
        )

    # =========================================================
    # VISION DETECTION
    # =========================================================

    def _detect_tables(
        self,
        image: Image.Image,
    ) -> List[Dict[str, Any]]:
        """
        Run the Table Transformer detector.

        Returns image-space boxes:

            {
                "bbox": [x0, y0, x1, y1],
                "score": 0.93,
                "label": "table"
            }
        """

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [image.size[::-1]],
            device=self.device,
        )

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

            score_value = float(
                score.detach().cpu().item()
            )

            label_id = int(
                label.detach().cpu().item()
            )

            bbox = [
                float(v)
                for v in box.detach().cpu().tolist()
            ]

            # TATR detection model uses:
            # 0 = table
            #
            # Keep this explicit rather than blindly accepting
            # every detected object.

            if label_id != 0:
                continue

            detections.append(
                {
                    "bbox": bbox,
                    "score": score_value,
                    "label": "table",
                }
            )

        detections = self._remove_duplicate_boxes(
            detections
        )

        detections.sort(
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            )
        )

        return detections

    # =========================================================
    # BOX UTILITIES
    # =========================================================

    @staticmethod
    def _box_iou(
        a: List[float],
        b: List[float],
    ) -> float:

        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0

        intersection = (
            (ix1 - ix0)
            * (iy1 - iy0)
        )

        area_a = max(
            (ax1 - ax0)
            * (ay1 - ay0),
            1e-6,
        )

        area_b = max(
            (bx1 - bx0)
            * (by1 - by0),
            1e-6,
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union

    @classmethod
    def _remove_duplicate_boxes(
        cls,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.80,
    ) -> List[Dict[str, Any]]:

        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda x: x["score"],
            reverse=True,
        )

        kept = []

        for candidate in detections:

            duplicate = False

            for existing in kept:

                iou = cls._box_iou(
                    candidate["bbox"],
                    existing["bbox"],
                )

                if iou >= iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(candidate)

        return kept

    @staticmethod
    def _clip_bbox(
        bbox: List[float],
        page,
        padding: float,
    ) -> List[float]:

        x0, y0, x1, y1 = bbox

        x0 -= padding
        y0 -= padding
        x1 += padding
        y1 += padding

        x0 = max(
            0.0,
            min(x0, page.width),
        )

        y0 = max(
            0.0,
            min(y0, page.height),
        )

        x1 = max(
            0.0,
            min(x1, page.width),
        )

        y1 = max(
            0.0,
            min(y1, page.height),
        )

        return [
            x0,
            y0,
            x1,
            y1,
        ]

    @classmethod
    def _valid_bbox(
        cls,
        bbox: List[float],
        page,
    ) -> bool:

        if not bbox or len(bbox) != 4:
            return False

        x0, y0, x1, y1 = bbox

        width = x1 - x0
        height = y1 - y0

        if width < cls.MIN_TABLE_WIDTH:
            return False

        if height < cls.MIN_TABLE_HEIGHT:
            return False

        if x1 <= x0:
            return False

        if y1 <= y0:
            return False

        if x1 <= 0:
            return False

        if y1 <= 0:
            return False

        if x0 >= page.width:
            return False

        if y0 >= page.height:
            return False

        return True

    # =========================================================
    # IMAGE → PDF COORDINATE CONVERSION
    # =========================================================

    @staticmethod
    def _image_bbox_to_pdf(
        image_bbox: List[float],
        image: Image.Image,
        page,
    ) -> List[float]:
        """
        Convert image pixel coordinates into pdfplumber
        PDF coordinates.

        Both systems use top-left origin for the values we
        use here, so this is primarily a scale conversion.
        """

        image_width, image_height = image.size

        scale_x = (
            page.width / image_width
        )

        scale_y = (
            page.height / image_height
        )

        x0, y0, x1, y1 = image_bbox

        return [
            x0 * scale_x,
            y0 * scale_y,
            x1 * scale_x,
            y1 * scale_y,
        ]

    # =========================================================
    # WORD EXTRACTION
    # =========================================================

    def _extract_words(
        self,
        cropped_page,
    ) -> List[Dict[str, Any]]:
        """
        Extract words from a confirmed table region.

        We deliberately do not use page.find_tables() here.

        The detector already told us this region is a table.
        """

        try:
            words = cropped_page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
                use_text_flow=False,
            )
        except Exception:
            return []

        cleaned = []

        for word in words:

            text = self._clean_text(
                word.get("text", "")
            )

            if not text:
                continue

            cleaned.append(
                {
                    "text": text,
                    "x0": float(
                        word.get("x0", 0)
                    ),
                    "x1": float(
                        word.get("x1", 0)
                    ),
                    "top": float(
                        word.get("top", 0)
                    ),
                    "bottom": float(
                        word.get("bottom", 0)
                    ),
                }
            )

        return cleaned

    # =========================================================
    # TABLE TEXT EXTRACTION (PURE SPATIAL)
    # =========================================================

    def _extract_table_text(
        self,
        page,
        bbox: List[float],
    ) -> List[List[str]]:
        """
        Extract table structure from a vision-confirmed
        table region by extracting words and clustering them
        spatially.
        """

        cropped = page.crop(
            tuple(bbox)
        )

        words = self._extract_words(
            cropped
        )

        return self._words_to_grid(
            words
        )

    @staticmethod
    def _words_to_grid(
        words: List[Dict[str, Any]],
        y_tolerance: float = 4.0,
        x_tolerance: float = 8.0,
    ) -> List[List[str]]:
        """
        Pure spatial word clustering.
        No heuristic grid inference.
        """
        if not words:
            return []

        # 1. Group by Y (Rows)
        words = sorted(
            words,
            key=lambda w: (
                w["top"],
                w["x0"],
            ),
        )

        rows = []
        for word in words:
            center_y = (word["top"] + word["bottom"]) / 2
            target = None

            # Look at the last few rows to see if we match
            for row in reversed(rows[-4:]):
                if abs(row["center_y"] - center_y) <= y_tolerance:
                    target = row
                    break

            if target is None:
                rows.append({
                    "center_y": center_y,
                    "words": [word],
                })
            else:
                target["words"].append(word)
                # update center_y moving average
                count = len(target["words"])
                target["center_y"] = ((target["center_y"] * (count - 1)) + center_y) / count

        # 2. Group by X within Rows (Columns)
        output_rows = []
        max_cols = 0
        
        for row in rows:
            row_words = sorted(row["words"], key=lambda w: w["x0"])
            cells = []
            current_cell_words = []
            
            for w in row_words:
                if not current_cell_words:
                    current_cell_words.append(w)
                else:
                    last_w = current_cell_words[-1]
                    # If horizontal gap is small, they belong to the same cell
                    if w["x0"] - last_w["x1"] <= x_tolerance:
                        current_cell_words.append(w)
                    else:
                        # Large gap -> new cell
                        text = " ".join(cw["text"] for cw in current_cell_words)
                        cells.append(text)
                        current_cell_words = [w]
                        
            if current_cell_words:
                text = " ".join(cw["text"] for cw in current_cell_words)
                cells.append(text)
                
            if cells:
                output_rows.append(cells)
                max_cols = max(max_cols, len(cells))
                
        # 3. Pad rows to uniform width
        for row in output_rows:
            while len(row) < max_cols:
                row.append("")
                
        return output_rows

    # =========================================================
    # POST-EXTRACTION CONTENT GATE
    # =========================================================

    @classmethod
    def _is_valid_table_content(
        cls,
        table: List[List[str]],
    ) -> bool:
        """
        Return True only when the extracted text actually looks
        like a table.  Called AFTER vision detection confirms a
        region — this filters out false positives that the model
        fired on (page numbers, figure captions, reference lists).

        Rules (all are rejection conditions):

        1. Single-cell table whose only text is a bare integer
           → page number / footer.

        2. All non-empty content starts with "Figure" or "Fig."
           → figure caption region.

        3. Majority of rows match a numbered-reference pattern
           ("N.  Author, ...") -> references section.

        4. Fewer than 2 distinct non-empty cells in the whole
           table → not enough content to be a real table.
        """

        non_empty = [
            c.strip()
            for row in table
            for c in row
            if c.strip()
        ]

        if not non_empty:
            return False

        # --- Rule 4: trivially empty --------------------------------
        if len(non_empty) < 2:
            # Allow single-cell only if it has substantial content
            if len(non_empty) == 1:
                text = non_empty[0]
                # Bare integer → page number
                if re.fullmatch(r"\d+", text):
                    return False
                # Very short → footer/label
                if len(text) < 8:
                    return False
            else:
                return False

        # --- Rule 1: bare page number --------------------------------
        # Single meaningful cell that is just a number
        if len(non_empty) == 1 and re.fullmatch(
            r"\d+", non_empty[0]
        ):
            return False

        # --- Rule 2: figure caption ----------------------------------
        figure_cells = sum(
            1 for c in non_empty
            if cls.FIGURE_RE.match(c)
        )
        if figure_cells / len(non_empty) > 0.5:
            return False

        # --- Rule 3: reference list ----------------------------------
        # Numbered reference lines look like: "1. Author, J., ..."
        # or "[1] Author" at the start of a row.
        ref_pattern = re.compile(
            r"^\s*(\[?\d+\]?\.?)\s+[A-Z]",
        )
        # Check row-level: join each row into one string, then test.
        ref_rows = 0
        for row in table:
            joined = " ".join(c for c in row if c.strip())
            if ref_pattern.match(joined):
                ref_rows += 1

        if len(table) > 0 and ref_rows / len(table) > 0.35:
            return False

        return True

    # =========================================================
    # CAPTION DETECTION
    # =========================================================

    def _get_page_lines(
        self,
        page,
    ) -> List[Dict[str, Any]]:

        try:
            words = page.extract_words(
                x_tolerance=2,
                y_tolerance=3,
                keep_blank_chars=False,
            )
        except Exception:
            return []

        if not words:
            return []

        words = sorted(
            words,
            key=lambda w: (
                float(
                    w.get("top", 0)
                ),
                float(
                    w.get("x0", 0)
                ),
            ),
        )

        lines = []

        for word in words:

            text = self._clean_text(
                word.get("text", "")
            )

            if not text:
                continue

            x0 = float(
                word.get("x0", 0)
            )

            x1 = float(
                word.get("x1", x0)
            )

            top = float(
                word.get("top", 0)
            )

            bottom = float(
                word.get("bottom", top)
            )

            target = None

            for line in reversed(
                lines[-5:]
            ):

                if abs(
                    line["top"]
                    - top
                ) <= 3:

                    target = line
                    break

            if target is None:

                lines.append(
                    {
                        "text": text,
                        "x0": x0,
                        "x1": x1,
                        "top": top,
                        "bottom": bottom,
                    }
                )

            else:

                target["text"] = (
                    target["text"]
                    + " "
                    + text
                )

                target["x0"] = min(
                    target["x0"],
                    x0,
                )

                target["x1"] = max(
                    target["x1"],
                    x1,
                )

                target["bottom"] = max(
                    target["bottom"],
                    bottom,
                )

        return lines

    def _find_caption(
        self,
        page,
        table_bbox: List[float],
    ) -> Tuple[
        Optional[str],
        Optional[List[float]],
    ]:
        """
        Find a caption near an already-confirmed table.

        IMPORTANT:

        Caption detection does not create a table.
        It only adds metadata to a table detected by vision.
        """

        lines = self._get_page_lines(
            page
        )

        if not lines:
            return None, None

        tx0, ty0, tx1, ty1 = (
            table_bbox
        )

        candidates = []

        for index, line in enumerate(
            lines
        ):

            if not self._is_caption(
                line["text"]
            ):
                continue

            caption = line["text"]

            caption_bbox = [
                line["x0"],
                line["top"],
                line["x1"],
                line["bottom"],
            ]

            # -------------------------------------------------
            # Caption above table
            # -------------------------------------------------

            if (
                line["bottom"]
                <= ty0
            ):

                vertical_distance = (
                    ty0
                    - line["bottom"]
                )

                horizontal_overlap = max(
                    0.0,
                    min(
                        tx1,
                        line["x1"],
                    )
                    - max(
                        tx0,
                        line["x0"],
                    ),
                )

                if (
                    vertical_distance
                    <= 100
                ):

                    score = (
                        vertical_distance
                        - horizontal_overlap
                    )

                    candidates.append(
                        (
                            score,
                            caption,
                            caption_bbox,
                        )
                    )

            # -------------------------------------------------
            # Caption below table
            # -------------------------------------------------

            elif (
                line["top"]
                >= ty1
            ):

                vertical_distance = (
                    line["top"]
                    - ty1
                )

                horizontal_overlap = max(
                    0.0,
                    min(
                        tx1,
                        line["x1"],
                    )
                    - max(
                        tx0,
                        line["x0"],
                    ),
                )

                if (
                    vertical_distance
                    <= 80
                ):

                    score = (
                        vertical_distance
                        - horizontal_overlap
                    )

                    candidates.append(
                        (
                            score,
                            caption,
                            caption_bbox,
                        )
                    )

        if not candidates:
            return None, None

        candidates.sort(
            key=lambda item: item[0]
        )

        _, caption, bbox = (
            candidates[0]
        )

        return caption, bbox

    # =========================================================
    # SAVE
    # =========================================================

    def _save_table(
        self,
        table: List[List[str]],
        page_number: int,
        table_index: int,
    ) -> str:

        filename = (
            f"table_p"
            f"{page_number}_"
            f"{table_index}.txt"
        )

        path = os.path.join(
            self.output_dir,
            filename,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            for row in table:

                f.write(
                    " | ".join(row)
                    + "\n"
                )

        return path

    # =========================================================
    # MAIN EXTRACTION
    # =========================================================

    def extract_tables(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Extract all vision-detected tables.

        No pdfplumber find_tables() call exists in this path.
        """

        results = []

        with pdfplumber.open(
            self.pdf_path
        ) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1,
            ):

                print(
                    f"[TableExtractor] "
                    f"Processing page "
                    f"{page_number}/{len(pdf.pages)}"
                )

                # -------------------------------------------------
                # 1. Render page
                # -------------------------------------------------

                try:

                    image = self._render_page(
                        page
                    )

                except Exception as exc:

                    print(
                        f"[WARNING] Could not render "
                        f"page {page_number}: {exc}"
                    )

                    continue

                # -------------------------------------------------
                # 2. Vision detection
                # -------------------------------------------------

                detections = (
                    self._detect_tables(
                        image
                    )
                )

                print(
                    f"[TableExtractor] "
                    f"Page {page_number}: "
                    f"{len(detections)} "
                    f"vision table(s)"
                )

                # -------------------------------------------------
                # 3. Convert boxes from image → PDF
                # -------------------------------------------------

                page_tables = []

                for detection in detections:

                    image_bbox = detection[
                        "bbox"
                    ]

                    pdf_bbox = (
                        self._image_bbox_to_pdf(
                            image_bbox,
                            image,
                            page,
                        )
                    )

                    pdf_bbox = (
                        self._clip_bbox(
                            pdf_bbox,
                            page,
                            self.padding,
                        )
                    )

                    if not self._valid_bbox(
                        pdf_bbox,
                        page,
                    ):
                        continue

                    # -------------------------------------------------
                    # 4. Extract text ONLY inside detector box
                    # -------------------------------------------------

                    table = (
                        self._extract_table_text(
                            page,
                            pdf_bbox,
                        )
                    )

                    if not table:
                        print(
                            f"[WARNING] Vision detected "
                            f"a table on page "
                            f"{page_number}, but no "
                            f"text could be extracted."
                        )
                        continue

                    # -------------------------------------------------
                    # 4b. Content gate: reject known false-positive
                    #     content patterns (page numbers, figure
                    #     captions, reference lists) even when the
                    #     vision model fired with confidence.
                    # -------------------------------------------------

                    if not self._is_valid_table_content(
                        table
                    ):
                        print(
                            f"[TableExtractor] "
                            f"Page {page_number}: "
                            f"rejected by content gate "
                            f"(confidence={detection['score']:.2f})"
                        )
                        continue

                    # -------------------------------------------------
                    # 5. Caption metadata
                    # -------------------------------------------------

                    caption, caption_bbox = (
                        self._find_caption(
                            page,
                            pdf_bbox,
                        )
                    )

                    page_tables.append(
                        {
                            "data": table,
                            "bbox": pdf_bbox,
                            "caption": caption,
                            "caption_bbox": caption_bbox,
                            "confidence": detection[
                                "score"
                            ],
                        }
                    )

                # -------------------------------------------------
                # 6. Remove duplicate detections
                # -------------------------------------------------

                unique_tables = []

                for candidate in page_tables:

                    duplicate = False

                    for existing in unique_tables:

                        if self._box_iou(
                            candidate["bbox"],
                            existing["bbox"],
                        ) >= 0.80:

                            duplicate = True

                            # Keep higher confidence.
                            if (
                                candidate[
                                    "confidence"
                                ]
                                > existing[
                                    "confidence"
                                ]
                            ):
                                existing.update(
                                    candidate
                                )

                            break

                    if not duplicate:
                        unique_tables.append(
                            candidate
                        )

                # -------------------------------------------------
                # 7. Reading order
                # -------------------------------------------------

                unique_tables.sort(
                    key=lambda item: (
                        item["bbox"][1],
                        item["bbox"][0],
                    )
                )

                # -------------------------------------------------
                # 8. Build final records
                # -------------------------------------------------

                for table_index, item in enumerate(
                    unique_tables,
                    start=1,
                ):

                    table = item["data"]

                    bbox = item["bbox"]

                    caption = item.get(
                        "caption"
                    )

                    caption_bbox = item.get(
                        "caption_bbox"
                    )

                    path = self._save_table(
                        table,
                        page_number,
                        table_index,
                    )

                    record = {
                        "table_id": (
                            f"table_p"
                            f"{page_number}_"
                            f"{table_index}"
                        ),
                        "page": page_number,
                        "bbox": bbox,
                        "caption": caption,
                        "caption_bbox": caption_bbox,
                        "table_path": path,
                        "data": table,

                        # Useful for debugging/model evaluation.
                        "detection_confidence": round(
                            float(
                                item[
                                    "confidence"
                                ]
                            ),
                            4,
                        ),
                    }

                    results.append(
                        record
                    )

        print(
            f"[TableExtractor] "
            f"Total tables detected: "
            f"{len(results)}"
        )

        return results

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def validate_tables(
        tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        errors = []
        warnings = []
        ids = set()

        if not isinstance(
            tables,
            list,
        ):
            return {
                "valid": False,
                "errors": [
                    "Tables output must be a list."
                ],
                "warnings": [],
                "table_count": 0,
            }

        for index, table in enumerate(
            tables
        ):

            prefix = (
                f"Table index {index}"
            )

            required_fields = [
                "table_id",
                "page",
                "bbox",
                "caption",
                "caption_bbox",
                "table_path",
                "data",
            ]

            for field in required_fields:

                if field not in table:

                    errors.append(
                        f"{prefix}: "
                        f"missing '{field}'"
                    )

            table_id = table.get(
                "table_id"
            )

            if table_id in ids:

                errors.append(
                    f"{prefix}: duplicate "
                    f"table_id "
                    f"'{table_id}'"
                )

            if table_id:
                ids.add(table_id)

            page = table.get(
                "page"
            )

            if (
                not isinstance(
                    page,
                    int,
                )
                or page < 1
            ):

                errors.append(
                    f"{prefix}: "
                    f"invalid page"
                )

            bbox = table.get(
                "bbox"
            )

            if (
                not isinstance(
                    bbox,
                    list,
                )
                or len(bbox) != 4
            ):

                errors.append(
                    f"{prefix}: "
                    f"invalid bbox"
                )

            data = table.get(
                "data"
            )

            if not data:

                errors.append(
                    f"{prefix}: "
                    f"empty table data"
                )

            elif not isinstance(
                data,
                list,
            ):

                errors.append(
                    f"{prefix}: "
                    f"data must be a list"
                )

            if not table.get(
                "caption"
            ):

                warnings.append(
                    f"{prefix}: "
                    f"no caption detected"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "table_count": len(tables),
        }


# =============================================================
# STANDALONE MAIN
# =============================================================

def main():
    """
    Standalone test entry point.

    You normally do NOT need this when calling the extractor
    from your project's main.py.
    """

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Vision-first PDF table extractor"
        )
    )

    parser.add_argument(
        "pdf",
        help="Path to PDF",
    )

    parser.add_argument(
        "--output-dir",
        default="output/tables",
        help="Directory for extracted tables",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.50,
        help="Vision detection confidence threshold",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, or leave empty for automatic",
    )

    args = parser.parse_args()

    extractor = TableExtractor(
        pdf_path=args.pdf,
        output_dir=args.output_dir,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    tables = extractor.extract_tables()

    validation = (
        extractor.validate_tables(
            tables
        )
    )

    print(
        json.dumps(
            {
                "tables": tables,
                "validation": validation,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()