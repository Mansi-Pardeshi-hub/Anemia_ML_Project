"""Execute this project's plain-Python notebook cells and save standard outputs.

This is a small fallback for environments without Jupyter/nbclient. It is
limited to the reviewed Member 4 analysis notebook; it is not a general-purpose
notebook executor. Run it from the project root with the project dependencies.
"""

import base64
import contextlib
import io
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "member4_ml_modelling" / "notebooks" / "member4_analysis.ipynb"


def main() -> None:
    os.chdir(ROOT)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            raise ValueError("This fallback only supports plain Python code cells.")

        execution_count += 1
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(source, f"{NOTEBOOK}:cell-{execution_count}", "exec"), namespace)

        outputs = []
        text = stdout.getvalue()
        if text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": text.splitlines(keepends=True),
            })

        pyplot = namespace.get("plt")
        if pyplot is not None:
            for number in pyplot.get_fignums():
                figure = pyplot.figure(number)
                image = io.BytesIO()
                figure.savefig(image, format="png", dpi=110, bbox_inches="tight")
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": base64.b64encode(image.getvalue()).decode("ascii")},
                    "metadata": {},
                })
            pyplot.close("all")

        cell["execution_count"] = execution_count
        cell["outputs"] = outputs

    notebook.setdefault("metadata", {}).setdefault(
        "kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}
    )
    notebook["nbformat"] = 4
    notebook.setdefault("nbformat_minor", 5)
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    temporary.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(NOTEBOOK)

    count = sum(cell.get("cell_type") == "code" for cell in notebook["cells"])
    with_outputs = sum(bool(cell.get("outputs")) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    image_count = sum(
        output.get("output_type") == "display_data"
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
    )
    print(f"Executed {count} plain-Python cells; {with_outputs} contain captured outputs and {image_count} contain figures.")
    print(f"Saved: {NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
