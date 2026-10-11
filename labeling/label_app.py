import json
import sys
from pathlib import Path

from nicegui import app, events, ui

# Make sibling modules (label_fields, card_type) importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from label_fields import FIELDS, handle_key, render_field  # noqa: E402

REVIEWED = "reviewed"  # name of the bool field that marks an image as done
fields_by_name = {f["name"]: f for f in FIELDS}

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


def label_path(name: str) -> Path:
    return labels_dir / f"{Path(name).stem}.json"


def read_label(name: str) -> dict:
    path = label_path(name)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def next_unreviewed(after: int):
    """Index of the next unreviewed image after `after` (wraps around), or None."""
    n = len(images)
    for i in list(range(after + 1, n)) + list(range(after)):
        if images[i] not in reviewed:
            return i
    return None


reviewed = {name for name in images if read_label(name).get(REVIEWED)}
first = next_unreviewed(-1)

# history holds the visited image indices, so "back" returns to what you saw before
state = {"history": [first if first is not None else 0], "pos": 0, "label": {}}


def current_name() -> str:
    return images[state["history"][state["pos"]]]


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

    def update_progress():
        progress.set_value(len(reviewed) / len(images))
        progress_label.set_text(f"{len(reviewed)} / {len(images)} reviewed")

    def set_value(field, value):
        state["label"][field["name"]] = value
        # Fields from other labelers stay in state["label"] and are written back untouched
        labels_dir.mkdir(exist_ok=True)
        label_path(current_name()).write_text(json.dumps(state["label"], indent=2), encoding="utf-8")

        if field["name"] == REVIEWED:
            (reviewed.add if value else reviewed.discard)(current_name())
            update_progress()
        # Text inputs keep their focus, everything else is redrawn
        if field["type"] in ("select", "bool"):
            fields.refresh()

    def show():
        name = current_name()
        state["label"] = read_label(name)
        header.set_text(name)
        image.set_source(f"/images/{name}")
        update_progress()
        fields.refresh()

    def go(delta):
        if delta < 0:
            state["pos"] = max(0, state["pos"] - 1)
        elif state["pos"] < len(state["history"]) - 1:
            state["pos"] += 1  # replay history after going back
        else:
            nxt = next_unreviewed(state["history"][state["pos"]])
            if nxt is None:
                ui.notify("No more unreviewed images")
                return
            state["history"].append(nxt)
            state["pos"] += 1
        show()

    def on_key(e: events.KeyEventArguments):
        if not e.action.keydown or e.modifiers.ctrl or e.modifiers.alt or e.modifiers.meta:
            return
        key = e.key.name.lower() if len(e.key.name) == 1 else e.key.name
        if key == "ArrowLeft":
            go(-1)
        elif key == "ArrowRight":
            go(1)
        elif e.action.repeat:
            return
        elif key == "Enter":
            # Mark as reviewed and continue with the next image
            set_value(fields_by_name[REVIEWED], True)
            go(1)
        else:
            for field in FIELDS:
                matched, value = handle_key(field, key, state["label"].get(field["name"]))
                if matched:
                    set_value(field, value)
                    return

    ui.keyboard(on_key=on_key)

    with ui.row().classes("w-full items-center no-wrap"):
        progress_label = ui.label().classes("text-gray-500 whitespace-nowrap")
        progress = ui.linear_progress(show_value=False).classes("grow")

    with ui.row().classes("w-full no-wrap"):
        with ui.column().classes("w-1/2"):
            header = ui.label().classes("text-gray-500")
            image = ui.image().props("fit=contain").classes("w-full h-[80vh]")
        with ui.column().classes("w-1/2 gap-4"):
            fields()

    show()


if __name__ == "__main__":
    ui.run(title="Label app", reload=False)