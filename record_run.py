"""
Records your real keyboard/mouse input (with timestamps) while you play
through a real attempt, so it can be converted into a seed genome for the
genetic algorithm - much more reliable than hand-describing a route.

Requires:
    pip install pynput

Usage:
    python record_run.py output.json

Start this BEFORE you begin the attempt (ideally right as the mission
starts / right after a restart), play through it normally (including
sprinting, aiming, shooting, using the weapon wheel), and press F12 when
you're done (e.g. right after exiting the level) to stop and save.

IMPORTANT CAVEAT: this reads the OS cursor's position, not the game's own
raw mouse input. Many first-person games capture/hide the cursor and read
raw relative deltas directly, bypassing normal OS cursor tracking - if
that's the case here, the recorded mouse "dx"/"dy" values for camera turns
may not correlate well with what actually happened in-game. Movement keys,
clicks, and the weapon wheel timing should still be captured reliably
either way. Do a short test recording first and sanity-check the mouse
values before trusting a full recording.
"""

import sys
import json
import time

from pynput import keyboard, mouse

STOP_KEY = keyboard.Key.f12

events = []
start_time = None


def now():
    return time.time() - start_time


def on_key_press(key):
    try:
        if key == STOP_KEY:
            listener_keyboard.stop()
            listener_mouse.stop()
            return
        name = key.char if hasattr(key, "char") and key.char else str(key).replace("Key.", "")
        events.append({"time": round(now(), 4), "type": "key_down", "key": name})
    except Exception:
        pass


def on_key_release(key):
    try:
        name = key.char if hasattr(key, "char") and key.char else str(key).replace("Key.", "")
        events.append({"time": round(now(), 4), "type": "key_up", "key": name})
    except Exception:
        pass


last_pos = {"x": None, "y": None}


def on_move(x, y):
    if last_pos["x"] is not None:
        dx = x - last_pos["x"]
        dy = y - last_pos["y"]
        # Skip no-op events (some OSes fire on_move even with zero delta).
        if dx != 0 or dy != 0:
            events.append({"time": round(now(), 4), "type": "mouse_move", "dx": dx, "dy": dy})
    last_pos["x"], last_pos["y"] = x, y


def on_click(x, y, button, pressed):
    btn_name = "left" if button == mouse.Button.left else "right" if button == mouse.Button.right else str(button)
    events.append({
        "time": round(now(), 4),
        "type": "mouse_down" if pressed else "mouse_up",
        "button": btn_name,
    })


def on_scroll(x, y, dx, dy):
    events.append({"time": round(now(), 4), "type": "scroll", "dy": dy})


def main():
    global start_time, listener_keyboard, listener_mouse

    if len(sys.argv) < 2:
        print("Usage: python record_run.py output.json")
        sys.exit(1)

    output_path = sys.argv[1]

    print("Recording will start in 3 seconds. Switch to the game now.")
    time.sleep(3)
    print("Recording! Play through your attempt now. Press F12 when done.")

    start_time = time.time()

    listener_keyboard = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener_mouse = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)

    listener_keyboard.start()
    listener_mouse.start()

    listener_keyboard.join()
    listener_mouse.join()

    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"\nSaved {len(events)} events to {output_path}")


if __name__ == "__main__":
    main()