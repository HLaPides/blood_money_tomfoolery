"""
Minimal isolated test: does pydirectinput arrow-key navigation actually
move the highlight in the pause menu?

Run this while in an active mission (not paused, not dead). It will open
the pause menu, then press Down twice. Watch whether the highlighted
option actually moves from Resume Game -> Options -> Restart.
"""

import time
import pydirectinput
from bloodmoney_env import BloodMoneyEnv, _send_extended_key

env = BloodMoneyEnv()

print("Focusing game window...")
env.focus_game_window()
time.sleep(1.0)

print("Pressing Escape in 2 seconds - watch the screen...")
time.sleep(2.0)
pydirectinput.press("esc")

print("Escape sent. Waiting 1s, then pressing Down twice, 0.3s apart...")
time.sleep(1.0)

_send_extended_key("down")
print("First Down sent - did the highlight move to Options?")
time.sleep(1.0)

_send_extended_key("down")
print("Second Down sent - did the highlight move to Restart?")
time.sleep(1.0)

print("Test done. Report exactly what you saw happen (or didn't).")