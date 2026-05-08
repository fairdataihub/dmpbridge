from pathlib import Path
import fitz

from dmpbridge.utils.logger import log


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def convert_pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 200
) -> list[Path]:
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    project_root = get_project_root()

    output_dir = (
        project_root
        / "data"
        / "page_images"
        / pdf_path.stem
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Converting PDF pages to images: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    image_paths = []

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    for page_index, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix)

        image_path = output_dir / f"page_{page_index}.png"
        pix.save(image_path)

        image_paths.append(image_path)

    doc.close()

    log(f"Saved {len(image_paths)} page images to: {output_dir}")

    return image_paths