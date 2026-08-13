"""
Hitman: Blood Money - Curtains Down bot environment.

Wraps memory reading (pymem) and input injection (pydirectinput) into a
simple reset() / step() / get_state() API, so a genetic algorithm (or any
other search method) doesn't need to know anything about the game itself.

Requires:
    pip install pymem pydirectinput

Run this while the game is running and loaded into the mission you want
to train on.
"""

import time as time_module
import pymem
import pydirectinput

# Disable pydirectinput's built-in pause between actions for speed.
pydirectinput.PAUSE = 0.0


# ---------------------------------------------------------------------------
# Memory addresses (confirmed via Cheat Engine, relative to exe base 0x400000)
# ---------------------------------------------------------------------------
PROCESS_NAME = "HitmanBloodMoney.exe"

OFFSET_LEVEL = 0x5B2550         # int, current level index
OFFSET_TIME = 0x5B25D4          # int, mission completion time in ms; 0 until finished
OFFSET_WITNESSES = 0x5B2574     # int, live witnesses; resets to 0 on mission restart
OFFSET_BODIES_FOUND = 0x5B2608  # int, bodies found; resets to 0 on mission restart
OFFSET_COVERS_BLOWN = 0x5B2614  # int, cover-blown count; resets to 0 on mission restart
OFFSET_TARGETS_KILLED = 0x5B25A4  # int, number of mission targets killed so far

# Curtains Down has 2 targets (Alvaro D'Alvade and Richard Delahunt).
TOTAL_TARGETS = 2

# How long to wait after sending Escape -> Enter for the mission to actually
# reset before we start reading state again. Tune this based on how long
# your restart transition actually takes.
RESTART_SETTLE_SECONDS = 2.0

# Safety timeout: if a single attempt runs this long without finishing,
# treat it as failed rather than waiting forever. This is also our
# fallback death/failure detection, since we couldn't pin down a reliable
# death flag - a real completion should take well under a minute (PB is
# 48s, category record is ~32s), so anything past this is either dead,
# stuck, or hopelessly off-route.
MAX_ATTEMPT_SECONDS = 75.0

# How often to poll game state while an attempt is running (seconds).
POLL_INTERVAL = 0.05


class BloodMoneyEnv:
    def __init__(self):
        self.pm = pymem.Pymem(PROCESS_NAME)
        self.base = self.pm.base_address

    # -- low-level memory helpers -----------------------------------------

    def _read_int(self, offset):
        return self.pm.read_int(self.base + offset)

    def get_state(self):
        """Snapshot of everything the fitness function needs."""
        return {
            "level": self._read_int(OFFSET_LEVEL),
            "time_ms": self._read_int(OFFSET_TIME),
            "witnesses": self._read_int(OFFSET_WITNESSES),
            "bodies_found": self._read_int(OFFSET_BODIES_FOUND),
            "covers_blown": self._read_int(OFFSET_COVERS_BLOWN),
            "targets_killed": self._read_int(OFFSET_TARGETS_KILLED),
        }

    def is_finished(self, state=None):
        state = state or self.get_state()
        return state["time_ms"] != 0

    def is_clean(self, state=None):
        """True if no SA-breaking event has happened (yet)."""
        state = state or self.get_state()
        return (
            state["witnesses"] == 0
            and state["bodies_found"] == 0
            and state["covers_blown"] == 0
        )

    # -- input ---------------------------------------------------------

    def send_key(self, key, duration=0.0):
        """Press and (optionally) hold a key for `duration` seconds."""
        pydirectinput.keyDown(key)
        if duration > 0:
            time_module.sleep(duration)
            pydirectinput.keyUp(key)
        else:
            pydirectinput.keyUp(key)

    def send_mouse_move(self, dx, dy):
        pydirectinput.moveRel(dx, dy, relative=True)

    def send_click(self, button="left", duration=0.0):
        pydirectinput.mouseDown(button=button)
        if duration > 0:
            time_module.sleep(duration)
        pydirectinput.mouseUp(button=button)

    def send_equip_weapon(self, angle_degrees, radius=150, hold_seconds=0.4):
        """
        Open the weapon wheel (hold right-click), move the mouse toward a
        given angle to hover over a wheel slot, hold briefly so the game
        registers the hover, then release to confirm the selection.

        angle_degrees: 0-360, direction to move the mouse from center.
        radius: how far (pixels) to move - tune against wheel sensitivity.
        hold_seconds: how long to hold after aiming, before releasing.
        """
        import math

        dx = int(radius * math.cos(math.radians(angle_degrees)))
        dy = int(radius * math.sin(math.radians(angle_degrees)))

        pydirectinput.mouseDown(button="right")
        pydirectinput.moveRel(dx, dy, relative=True)
        time_module.sleep(hold_seconds)
        pydirectinput.mouseUp(button="right")

    # -- lifecycle -------------------------------------------------------

    def reset(self):
        """
        Restart the current mission attempt.
        Covers both cases: death screen (Restart already focused) and
        pause menu (Escape opens it, Restart is the first option).
        """
        pydirectinput.press("esc")
        time_module.sleep(0.3)
        pydirectinput.press("enter")
        time_module.sleep(RESTART_SETTLE_SECONDS)

        state = self.get_state()
        # Sanity check: everything should be back to zero after a reset.
        if not self.is_clean(state) or state["time_ms"] != 0:
            # Give it a bit more time in case the transition was slow.
            time_module.sleep(1.0)
            state = self.get_state()

        return state

    def run_attempt(self, action_sequence):
        """
        Play back a full list of timed actions, then wait for the mission
        to finish (or time out), and return the final state.

        action_sequence: list of (delay_seconds, callable) tuples, where
        `callable` takes no arguments and performs one input action, e.g.:
            (0.0,  lambda: env.send_key("w", duration=0.5))
            (0.5,  lambda: env.send_mouse_move(50, 0))
        Delays are relative to the start of the attempt, not to each other.
        """
        start = time_module.time()
        idx = 0
        n = len(action_sequence)

        while True:
            elapsed = time_module.time() - start

            # Fire any actions whose scheduled time has arrived.
            while idx < n and action_sequence[idx][0] <= elapsed:
                _, action_fn = action_sequence[idx]
                action_fn()
                idx += 1

            state = self.get_state()
            if self.is_finished(state):
                return state

            if elapsed > MAX_ATTEMPT_SECONDS:
                # Attempt got stuck / never finished - treat as a failure.
                return state

            time_module.sleep(POLL_INTERVAL)

    def score(self, state):
        """
        Lower is better. Three tiers, worst to best:

          3. Didn't finish        - scored by how much progress was made
                                     (targets killed so far), so the search
                                     has a gradient to climb even before
                                     anything ever completes the mission.
          2. Finished, but broke  - better than not finishing at all, but
             SA at some point       worse than any clean finish. Ranked by
                                     how minor the break was.
          1. Finished, clean SA   - ranked purely by time. Lower is faster.

        The tier gaps are large fixed constants so a run in a better tier
        always beats every run in a worse tier, regardless of the in-tier
        details - but within a tier, runs are ranked by real progress/time,
        not treated as identical.
        """
        TIER_UNFINISHED = 2_000_000
        TIER_FINISHED_DIRTY = 1_000_000

        targets_killed = state["targets_killed"]

        if not self.is_finished(state):
            # More targets killed = closer to actually finishing = better,
            # even though it's still in the worst tier overall.
            progress_credit = targets_killed * 100_000
            return TIER_UNFINISHED - progress_credit

        if not self.is_clean(state):
            # Finished, but broke SA. Rank by how minor the break was -
            # fewer/lighter incidents score better within this tier.
            break_penalty = (
                state["witnesses"] * 100
                + state["bodies_found"] * 200
                + state["covers_blown"] * 150
            )
            return TIER_FINISHED_DIRTY + break_penalty

        # Clean, finished run - score is just the time. Lower is faster.
        return state["time_ms"]


if __name__ == "__main__":
    # Manual test: connect, then poll until the mission actually finishes.
    # No timeout - play through a full attempt (finish, die, whatever) at
    # your own pace. Ctrl+C to stop early if needed.
    env = BloodMoneyEnv()
    print("Connected. Play through a full attempt now.")
    print("Polling until the mission finishes (Ctrl+C to stop early)...\n")

    last_state = None
    try:
        while True:
            state = env.get_state()
            if state != last_state:
                print(state)
                last_state = state
            if env.is_finished(state):
                print("\nMission ended! Final state above.")
                break
            time_module.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped by user.")