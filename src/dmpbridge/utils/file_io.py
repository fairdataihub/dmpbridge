from pathlib import Path
import json


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict | list, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(input_path: str | Path) -> dict | list:
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(text: str, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        
def load_text(path: str | Path) -> str:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()