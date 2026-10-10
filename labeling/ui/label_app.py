import json
import sys
from pathlib import Path

from nicegui import app, events, ui

# Make sibling modules (label_fields, card_type) importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.label_fields import FIELDS, handle_key, render_field  # noqa: E402

if len(sys.argv) < 2:
    sys.exit("Usage: python label_app.py BASE_DIR")

base_dir = Path(sys.argv[1])
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

if not images_dir.is_dir():
    sys.exit(f"Error: '{images_dir}' does not exist.")
images = sorted(p.name for p in images_dir.iterdir() if p.suffix.lower() == ".jpg")
if not images:
    sys.exit("No .jpg files found.")

app.add_static_files("/images", images_dir)

state = {"index": 0, "label": {}}


def label_path() -> Path:
    return labels_dir / f"{Path(images[state['index']]).stem}.json"


def load():
    path = label_path()
    state["label"] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save():
    # Fields from other labelers stay in state["label"] and are written back untouched
    labels_dir.mkdir(exist_ok=True)
    label_path().write_text(json.dumps(state["label"], indent=2), encoding="utf-8")


@ui.page("/")
def index():
    @ui.refreshable
    def fields():
        for field in FIELDS:
            value = state["label"].get(field["name"])
            render_field(field, value, lambda v, f=field: set_value(f, v))

        # Fields that are not configured in FIELDS (read-only)
        known = {f["name"] for f in FIELDS}
        others = {k: v for k, v in state["label"].items() if k not in known}
        if others:
            ui.code(json.dumps(others, indent=2), language="json")

    def set_value(field, value):
        state["label"][field["name"]] = value
        save()
        # Text inputs keep their focus, everything else is redrawn
        if field["type"] in ("select", "bool"):
            fields.refresh()

    def show():
        load()
        name = images[state["index"]]
        header.set_text(f"{state['index'] + 1} / {len(images)} - {name}")
        image.set_source(f"/images/{name}")
        fields.refresh()

    def go(delta):
        state["index"] = max(0, min(state["index"] + delta, len(images) - 1))
        show()

    def on_key(e: events.KeyEventArguments):
        if not e.action.keydown or e.modifiers.ctrl or e.modifiers.alt or e.modifiers.meta:
            return
        key = e.key.name.lower() if len(e.key.name) == 1 else e.key.name
        if key == "ArrowLeft":
            go(-1)
        elif key == "ArrowRight":
            go(1)
        elif not e.action.repeat:
            for field in FIELDS:
                matched, value = handle_key(field, key, state["label"].get(field["name"]))
                if matched:
                    set_value(field, value)
                    return

    ui.keyboard(on_key=on_key)

    with ui.row().classes("w-full no-wrap"):
        with ui.column().classes("w-1/2"):
            header = ui.label().classes("text-gray-500")
            image = ui.image().props("fit=contain").classes("w-full h-[85vh]")
        with ui.column().classes("w-1/2 gap-4"):
            fields()

    show()


if __name__ == "__main__":
    ui.run(title="Label app", reload=False)