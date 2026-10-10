from nicegui import ui

from prelabel.labeler.ygo_card_type import COLORS

# One entry per label field.
# type: "select" (needs "options", optional "keys" {key: option}),
#       "text", "number" or "bool" (optional "key" toggles it)
FIELDS = [
    {
        "name": "card_type",
        "type": "select",
        # "pendulum" is set by the labeler but is not a frame color
        "options": list(COLORS) + ["pendulum"],
        "keys": {
            "n": "normal", "e": "effect", "r": "ritual", "f": "fusion", "y": "synchro",
            "x": "xyz", "l": "link", "s": "spell", "t": "trap", "p": "pendulum",
        },
    },
    {"name": "faulty", "type": "bool", "key": "d"},
    {"name": "reviewed", "type": "bool", "key": "Enter"},
]

CHIP = "px-3 py-1 rounded-full cursor-pointer select-none "
CHIP_ON = CHIP + "bg-blue-600 text-white"
CHIP_OFF = CHIP + "bg-gray-200 text-gray-800"


def handle_key(field, key, value):
    """Return (True, new_value) if the key is a shortcut of this field, else (False, value)."""
    if field["type"] == "select" and key in field.get("keys", {}):
        return True, field["keys"][key]
    if field["type"] == "bool" and key == field.get("key"):
        return True, not value
    return False, value


def render_field(field, value, on_change):
    """Draw the widget for one field. on_change(new_value) is called when the user edits it."""
    name, kind = field["name"], field["type"]

    # Selects and bools are plain labels (not focusable), so keyboard shortcuts keep working
    if kind == "select":
        hints = {option: key for key, option in field.get("keys", {}).items()}
        ui.label(name).classes("text-sm text-gray-500")
        with ui.row().classes("gap-2"):
            for option in field["options"]:
                text = f"[{hints[option]}] {option}" if option in hints else option
                ui.label(text).classes(CHIP_ON if option == value else CHIP_OFF).on(
                    "click", lambda _, o=option: on_change(o)
                )
    elif kind == "bool":
        hint = f" [{field['key']}]" if "key" in field else ""
        ui.label(f"{'✓' if value else '✗'} {name}{hint}").classes(
            CHIP_ON if value else CHIP_OFF
        ).on("click", lambda _: on_change(not value))
    elif kind == "text":
        # Enter leaves the input so the shortcuts work again
        ui.input(name, value=value or "", on_change=lambda e: on_change(e.value)).on(
            "keydown.enter", lambda _: ui.run_javascript("document.activeElement.blur()")
        )
    elif kind == "number":
        ui.number(name, value=value, on_change=lambda e: on_change(e.value))
    else:
        raise ValueError(f"Unknown field type '{kind}' for field '{name}'")