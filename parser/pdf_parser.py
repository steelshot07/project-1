import re
import pymupdf as fitz


class PDFParser:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    # ---------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------

    def extract_metadata(self):
        meta = self.doc.metadata

        return {
            "title": meta.get("title") or "Unknown Title",
            "authors": meta.get("author") or "Unknown Author",
            "total_pages": len(self.doc),
        }

    # ---------------------------------------------------------
    # HEADING DETECTION
    # ---------------------------------------------------------

    def is_heading(self, block):
        """
        Detect headings using font size, boldness and numbering.
        """

        if not block.get("lines"):
            return False

        spans = []

        for line in block["lines"]:
            for span in line["spans"]:
                spans.append(span)

        if not spans:
            return False

        text = " ".join(
            span["text"].strip() for span in spans if span["text"].strip()
        ).strip()

        if not text:
            return False

        # Ignore very long text blocks
        if len(text) > 150:
            return False

        max_font_size = max(span["size"] for span in spans)

        bold = any("bold" in span["font"].lower() for span in spans)

        # Numbered headings:
        # 1 Introduction
        # 2.1 Methodology
        # 3.2.1 Dataset
        numbered_heading = re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text)

        # Common academic headings
        known_heading = re.match(
            r"^(abstract|introduction|background|"
            r"related work|methodology|methods|"
            r"experiments|results|discussion|"
            r"conclusion|references|acknowledgements?)$",
            text,
            re.IGNORECASE,
        )

        # Large/bold text is likely a heading
        large_text = max_font_size >= 13

        return (
            numbered_heading is not None
            or known_heading is not None
            or (bold and large_text)
        )

    # ---------------------------------------------------------
    # HEADING LEVEL
    # ---------------------------------------------------------

    def get_heading_level(self, text):

        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", text)

        if match:
            numbering = match.group(1)
            return numbering.count(".") + 1

        # Non-numbered major sections
        if text.lower() in [
            "abstract",
            "introduction",
            "background",
            "related work",
            "methodology",
            "methods",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "references",
        ]:
            return 1

        return 1

    # ---------------------------------------------------------
    # BLOCK TYPE
    # ---------------------------------------------------------

    def classify_block(self, block):

        text = block["text"].strip()

        if not text:
            return "empty"

        if self.is_heading(block["raw"]):
            return "heading"

        if re.match(r"^(figure|fig\.?)\s*\d+", text, re.IGNORECASE):
            return "figure_caption"

        if re.match(r"^table\s*\d+", text, re.IGNORECASE):
            return "table_caption"

        if re.match(r"^(equation|eq\.?)\s*\(?\d+\)?", text, re.IGNORECASE):
            return "equation"

        return "paragraph"

    # ---------------------------------------------------------
    # EXTRACT RAW BLOCKS
    # ---------------------------------------------------------

    def extract_page_blocks(self, page):

        page_dict = page.get_text("dict")

        blocks = []

        for index, raw_block in enumerate(page_dict["blocks"]):

            # Ignore images at this stage
            if raw_block["type"] != 0:
                continue

            lines = raw_block.get("lines", [])

            text_parts = []

            for line in lines:
                for span in line.get("spans", []):
                    text_parts.append(span["text"])

            text = " ".join(text_parts).strip()

            if not text:
                continue

            bbox = raw_block["bbox"]

            block = {
                "block_id": f"p{page.number + 1}_b{index}",
                "page": page.number + 1,
                "bbox": [round(x, 2) for x in bbox],
                "text": text,
                "raw": raw_block,
            }

            block["type"] = self.classify_block(block)

            blocks.append(block)

        return blocks

    # ---------------------------------------------------------
    # BUILD SECTIONS
    # ---------------------------------------------------------

    def build_sections(self, blocks):

        sections = []

        current_section = None
        current_subsection = None

        for block in blocks:

            if block["type"] == "heading":

                level = self.get_heading_level(block["text"])

                section = {
                    "section_id": f"section_{len(sections) + 1}",
                    "title": block["text"],
                    "level": level,
                    "page": block["page"],
                    "blocks": [],
                }

                if level == 1:

                    sections.append(section)

                    current_section = section
                    current_subsection = None

                else:

                    if current_section is None:
                        current_section = {
                            "section_id": "section_0",
                            "title": "Unknown",
                            "level": 1,
                            "page": block["page"],
                            "blocks": [],
                            "subsections": [],
                        }

                        sections.append(current_section)

                    if "subsections" not in current_section:
                        current_section["subsections"] = []

                    current_section["subsections"].append(section)

                    current_subsection = section

                continue

            # Normal semantic block

            clean_block = {
                "block_id": block["block_id"],
                "page": block["page"],
                "bbox": block["bbox"],
                "type": block["type"],
                "text": block["text"],
            }

            if current_subsection is not None:
                current_subsection["blocks"].append(clean_block)

            elif current_section is not None:
                current_section["blocks"].append(clean_block)

        return sections

    # ---------------------------------------------------------
    # PAGE EXTRACTION
    # ---------------------------------------------------------

    def extract_pages(self):

        pages = []

        for page in self.doc:

            blocks = self.extract_page_blocks(page)

            pages.append(
                {
                    "page_number": page.number + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "blocks": [
                        {
                            "block_id": b["block_id"],
                            "bbox": b["bbox"],
                            "type": b["type"],
                            "text": b["text"],
                        }
                        for b in blocks
                    ],
                }
            )

        return pages

    # ---------------------------------------------------------
    # COMPLETE DOCUMENT
    # ---------------------------------------------------------

    def parse(self):

        all_blocks = []

        for page in self.doc:
            blocks = self.extract_page_blocks(page)
            all_blocks.extend(blocks)

        pages = self.extract_pages()
        sections = self.build_sections(all_blocks)

        return {
            "metadata": self.extract_metadata(),
            "pages": pages,
            "sections": sections,
        }

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    def validate(self, parsed):

        errors = []

        if "metadata" not in parsed:
            errors.append("Missing metadata")

        if "pages" not in parsed:
            errors.append("Missing pages")

        if "sections" not in parsed:
            errors.append("Missing sections")

        for page in parsed.get("pages", []):

            if "page_number" not in page:
                errors.append("Page missing page_number")

            if "blocks" not in page:
                errors.append(f"Page {page.get('page_number')} missing blocks")

        return {"valid": len(errors) == 0, "errors": errors}

    def close(self):
        self.doc.close()
