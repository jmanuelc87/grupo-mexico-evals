"""Pipeline scaffolding: PDFs -> pages -> layout detection -> region extraction -> (markdown, TBD)."""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import sys
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import easyocr
import numpy as np
import pypdfium2 as pdfium
import torch
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor
from transformers.utils import logging as hf_logging
from transformers.models.detr import DetrForSegmentation

LAYOUT_MODEL = "cmarkea/detr-layout-detection"
DEFAULT_THRESHOLD = 0.4
RENDER_SCALE = 2.0  # 72 dpi * 2
OCR_LANGUAGES = ["es", "en"]
VISION_MODEL = "claude-haiku-5-5"
# Layout classes whose content is plain text, handled by OCR.
OCR_LABELS = frozenset(
    {
        "Caption",
        "Footnote",
        "List-item",
        "Page-footer",
        "Page-header",
        "Section-header",
        "Text",
        "Title",
    }
)

# Layout classes OCR can't handle, extracted by a vision model instead.
VISION_PROMPTS = {
    "Picture": "Describe this image concisely: what it shows and any text or data it conveys.",
    "Table": "Extract this table as a GitHub-flavored markdown table. Output only the table.",
    "Formula": "Transcribe this math formula as plain text (no LaTeX). Output only the formula.",
}
VISION_SYSTEM_PROMPT = (
    "You extract content from a cropped region of a document page. "
    "Answer with the requested content only, no preamble."
)


@dataclass
class Page:
    """A single rendered PDF page."""

    source: Path
    number: int  # 1-based
    image: Image.Image


@dataclass
class LayoutBox:
    label: str
    score: float
    bbox: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (pixels)
    text: str | None = None  # filled by the OCR / vision step


@dataclass
class PageLayout:
    page: Page
    boxes: list[LayoutBox] = field(default_factory=list)


def list_pdfs(folder: Path) -> list[Path]:
    """Step 1: every PDF in the folder."""
    return sorted(folder.glob("*.pdf"))


def split_pages(pdf_path: Path, scale: float = RENDER_SCALE) -> Iterator[Page]:
    """Step 2: split a PDF into rendered pages."""
    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for index in range(len(pdf)):
            image = pdf[index].render(scale=scale).to_pil().convert("RGB")
            yield Page(source=pdf_path, number=index + 1, image=image)
    finally:
        pdf.close()


def count_pages(folder: Path) -> int:
    """Total page count across every PDF in the folder, for the progress bar."""
    total = 0
    for pdf_path in list_pdfs(folder):
        pdf = pdfium.PdfDocument(pdf_path)
        try:
            total += len(pdf)
        finally:
            pdf.close()
    return total


def pick_device() -> str:
    """CUDA if present, else Apple Metal (MPS) on macOS, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LayoutDetector:
    """Step 3: layout detection with cmarkea/detr-layout-detection."""

    def __init__(
        self,
        model_name: str = LAYOUT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self.threshold = threshold
        self.device = device or pick_device()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DetrForSegmentation.from_pretrained(model_name).to(self.device).eval()
        self.id2label: dict[int, str] = self.model.config.id2label

    @torch.inference_mode()
    def detect(self, page: Page) -> PageLayout:
        image = page.image
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model(**inputs)
        results = self.processor.post_process_object_detection(
            output,
            threshold=self.threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = [
            LayoutBox(
                label=self.id2label[int(label)],
                score=float(score),
                bbox=tuple(float(v) for v in box),  # type: ignore[arg-type]
            )
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            )
        ]
        return PageLayout(page=page, boxes=boxes)


def crop_region(page: Page, box: LayoutBox) -> Image.Image:
    """Crop a detected region out of its page image."""
    width, height = page.image.size
    x_min, y_min, x_max, y_max = box.bbox
    return page.image.crop(
        (
            max(0, int(x_min)),
            max(0, int(y_min)),
            min(width, int(x_max) + 1),
            min(height, int(y_max) + 1),
        )
    )


class TextOCR:
    """Step 4: OCR of the text regions with EasyOCR."""

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool | None = None,
    ) -> None:
        self.reader = easyocr.Reader(
            languages or OCR_LANGUAGES,
            gpu=torch.cuda.is_available() if use_gpu is None else use_gpu,
        )

    def read(self, image: Image.Image) -> str:
        # paragraph=True merges lines into blocks in reading order
        lines = self.reader.readtext(np.asarray(image), detail=0, paragraph=True)
        return "\n".join(lines)

    def run(self, layout: PageLayout) -> PageLayout:
        """OCR every text-like region of the page, in place."""
        for box in layout.boxes:
            if box.label in OCR_LABELS:
                box.text = self.read(crop_region(layout.page, box))
        return layout


class VisionExtractor:
    """Step 4b: Picture / Table / Formula extraction with Claude (Agent SDK)."""

    def __init__(self, model: str = VISION_MODEL) -> None:
        self.options = ClaudeAgentOptions(
            model=model,
            system_prompt=VISION_SYSTEM_PROMPT,
            tools=[],  # no built-in tools: pure vision call
            max_turns=1,
            setting_sources=[],  # ignore user/project settings and CLAUDE.md
        )

    async def _extract(self, image: Image.Image, label: str) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(buffer.getvalue()).decode(),
                },
            },
            {"type": "text", "text": VISION_PROMPTS[label]},
        ]

        async def prompt():
            yield {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
            }

        parts: list[str] = []
        async for message in query(prompt=prompt(), options=self.options):
            if isinstance(message, AssistantMessage):
                parts.extend(b.text for b in message.content if isinstance(b, TextBlock))
        return "".join(parts).strip()

    async def _run(self, layout: PageLayout) -> None:
        for box in layout.boxes:
            if box.label in VISION_PROMPTS:
                box.text = await self._extract(crop_region(layout.page, box), box.label)

    def run(self, layout: PageLayout) -> PageLayout:
        """Extract every Picture / Table / Formula region of the page, in place."""
        asyncio.run(self._run(layout))
        return layout


def run(
    input_dir: Path,
    output_dir: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    scale: float = RENDER_SCALE,
) -> Iterator[PageLayout]:
    """Run the pipeline over every PDF in ``input_dir``.

    If ``output_dir`` is set, each page is also written there as
    ``<pdf-stem>_p<page>.txt``, with the extracted region text in reading order.
    """
    detector = LayoutDetector(threshold=threshold)
    ocr = TextOCR()
    vision = VisionExtractor()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in list_pdfs(input_dir):
        for page in split_pages(pdf_path, scale=scale):
            layout = vision.run(ocr.run(detector.detect(page)))
            if output_dir is not None:
                write_markdown(layout, output_dir)
            yield layout


def write_markdown(layout: PageLayout, output_dir: Path) -> Path:
    """Write the extracted text of one page to ``output_dir``."""
    page = layout.page
    path = output_dir / f"{page.source.stem}_p{page.number:03d}.txt"
    blocks = [box.text for box in layout.boxes if box.text]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect layout regions in PDF pages.")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="folder containing the PDFs (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="minimum detection score to keep a region (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=RENDER_SCALE,
        help="render scale for PDF pages, 1.0 = 72 dpi (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="folder to write one markdown file per page (default: none, print only)",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    """Send logging and Python warnings to stderr, keeping stdout for results."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    # Known noise from EasyOCR / PyTorch that we can't fix from here.
    # Filters added later take precedence over the "default" rule above.
    warnings.filterwarnings("ignore", message=r"torch\.ao\.quantization is deprecated")
    warnings.filterwarnings("ignore", message=r"torch\.quantize_per_tensor")
    warnings.filterwarnings("ignore", message=r"'pin_memory' argument is set as true")
    logging.getLogger("easyocr").setLevel(logging.ERROR)  # silences "Using CPU"
    hf_logging.disable_progress_bar()  # silences the "Loading weights" bar


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"input folder not found: {args.input_dir}")

    results = run(
        args.input_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        scale=args.scale,
    )
    with tqdm(total=count_pages(args.input_dir), desc="Pages", unit="page", file=sys.stdout) as progress:
        for result in results:
            progress.update(1)
            if args.output_dir is None:
                print(f"{result.page.source.name} p{result.page.number}: {len(result.boxes)} regions")
                for box in result.boxes:
                    if box.text:
                        print(f"  [{box.label}] {box.text!r}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
