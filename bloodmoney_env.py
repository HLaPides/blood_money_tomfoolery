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
import ctypes
import pymem
import pydirectinput
import win32gui
import win32process
import win32con
import win32api

# Disable pydirectinput's built-in pause between actions for speed.
pydirectinput.PAUSE = 0.0

# Disable the corner-based fail-safe: it assumes runaway absolute mouse
# movement is a bug, but our relative look_turn moves can legitimately
# drift the real OS cursor into a corner over a long session even though
# the game reads raw input deltas and doesn't care about absolute cursor
# position. We rely on our own crash-safety (release_all_inputs) instead.
pydirectinput.FAILSAFE = False

MOUSEEVENTF_WHEEL = 0x0800
WHEEL_DELTA = 120


def _scroll_wheel(clicks):
    """
    pydirectinput doesn't expose a scroll function in this version, so we
    call the underlying Windows API directly instead.
    """
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(clicks * WHEEL_DELTA), 0)


# --- Extended-key (arrow key) sender ---------------------------------
# Arrow keys are "extended keys" at the scan-code level - they share the
# same base scan code as the numpad digits, and are only distinguished by
# a special extended-key flag. pydirectinput.press("up"/"down"/"left")
# doesn't reliably set this flag, so the game may be reading these as
# numpad input instead of real arrow keys and silently ignoring them.
# We bypass pydirectinput here and send proper scan codes directly.

PUL = ctypes.POINTER(ctypes.c_ulong)


class _KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class _InputUnion(ctypes.Union):
    _fields_ = [("ki", _KeyBdInput),
                ("mi", _MouseInput),
                ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", _InputUnion)]


_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008

# Standard Set-1 scan codes (US QWERTY). Arrow keys are "extended" keys -
# they share a base scan code with the numpad and need the extended-key
# flag set, or the game may read them as numpad input and ignore them
# (this is what silently broke arrow-key menu navigation earlier).
# Enter/Escape don't need the extended flag - these are the "main" keys,
# not their numpad counterparts.
_SCAN_CODES = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "e": 0x12, "g": 0x22,
    "shift": 0x2A, "ctrl": 0x1D,
    "enter": 0x1C, "esc": 0x01,
    "up": 0x48, "down": 0x50, "left": 0x4B, "right": 0x4D,
}

_EXTENDED_KEYS = {"up", "down", "left", "right"}


def _scancode_key_down(name):
    scan_code = _SCAN_CODES[name]
    flags = _KEYEVENTF_SCANCODE
    if name in _EXTENDED_KEYS:
        flags |= _KEYEVENTF_EXTENDEDKEY
    extra = ctypes.c_ulong(0)
    inp = _Input(_INPUT_KEYBOARD, _InputUnion(ki=_KeyBdInput(
        0, scan_code, flags, 0, ctypes.pointer(extra)
    )))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


def _scancode_key_up(name):
    scan_code = _SCAN_CODES[name]
    flags = _KEYEVENTF_SCANCODE | _KEYEVENTF_KEYUP
    if name in _EXTENDED_KEYS:
        flags |= _KEYEVENTF_EXTENDEDKEY
    extra = ctypes.c_ulong(0)
    inp = _Input(_INPUT_KEYBOARD, _InputUnion(ki=_KeyBdInput(
        0, scan_code, flags, 0, ctypes.pointer(extra)
    )))
    ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))


def _scancode_press(name, hold_seconds=0.05):
    """Press and release a key using its real scan code, via the Windows API directly."""
    _scancode_key_down(name)
    time_module.sleep(hold_seconds)
    _scancode_key_up(name)


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

# Player position pointer chain, reverse-engineered from the open-source
# 0xvpr/HM3-Trainer teleport hack (offsets.hpp / hacks.cpp / memory.hpp).
# Resolution: start at exe_base + PLAYER_XYZ_BASE, then for each offset in
# PLAYER_XYZ_OFFSETS: dereference the current address, then add the offset.
# The final address holds x (float), with y and z immediately following.
PLAYER_XYZ_BASE = 0x41F83C
PLAYER_XYZ_OFFSETS = [0xA20, 0x4, 0x50, 0x24]

# Entity list, reverse-engineered from the same trainer source. A single
# dereference at this address gives an EntityList* (n_entities count,
# followed by an array of Entity* pointers). Each Entity has x/y/z floats
# at this fixed offset. Shares the same static base address as the player
# pointer above, but resolved with a single dereference instead of the
# longer offset chain.
ENTITY_LIST_BASE = 0x41F83C
ENTITY_XYZ_OFFSET = 0x3DC

# Curtains Down has 2 targets (Alvaro D'Alvade and Richard Delahunt).
TOTAL_TARGETS = 2

# Fill this in with your real vantage-point coordinates - grab it with
# test_position.py while standing where you actually shoot both targets.
# Format: (x, y, z). Leave as None to disable distance-based scoring
# entirely (falls back to the old kills-only behavior for that tier).
VANTAGE_POINT = (465.093994140625, -644.819091796875, 1014.2216186523438)

# Safety cap on the distance tie-breaker in score() - guarantees a clean
# (undetected) unfinished run can never mathematically score worse than
# a dirty (detected) unfinished run, no matter how large a real distance
# value turns out to be.
MAX_DISTANCE_CREDIT = 500_000

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
        self.hwnd = None
        self.focus_game_window()

    def _recenter_cursor(self):
        """Move the real OS cursor back to the center of the screen, so it can't drift toward an edge over a long session."""
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
            pydirectinput.moveTo(screen_w // 2, screen_h // 2)
        except Exception:
            pass

    def focus_game_window(self):
        """
        Find the game's window by matching its owning process ID (which we
        already know exactly, from pymem) rather than guessing its title
        text - titles can vary with fullscreen/borderless mode, but the
        process ID is unambiguous. Bring it to the foreground so
        pydirectinput's keystrokes/clicks actually reach the game instead
        of whatever window (e.g. this terminal) currently has focus.
        """
        target_pid = self.pm.process_id

        def _callback(hwnd, results):
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid:
                results.append(hwnd)

        results = []
        win32gui.EnumWindows(_callback, results)

        if not results:
            print(f"WARNING: couldn't find a visible window for PID {target_pid}. "
                  f"Inputs may not reach the game.")
            return False

        self.hwnd = results[0]

        # Restore if minimized.
        win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)

        # Windows blocks background processes from stealing foreground
        # focus outright. AttachThreadInput lets our thread temporarily
        # share input state with the target window's thread, which is
        # enough to satisfy SetForegroundWindow - unlike the old Alt-key-
        # press workaround, this never simulates any real keyboard input,
        # so it can't be misread as part of an Alt-Tab gesture.
        try:
            current_thread_id = win32api.GetCurrentThreadId()
            target_thread_id, _ = win32process.GetWindowThreadProcessId(self.hwnd)

            win32process.AttachThreadInput(target_thread_id, current_thread_id, True)
            try:
                win32gui.SetForegroundWindow(self.hwnd)
            finally:
                win32process.AttachThreadInput(target_thread_id, current_thread_id, False)
        except Exception as e:
            print(f"WARNING: couldn't force focus ({e}). "
                  f"Inputs may not reach the game this attempt.")
            return False

        time_module.sleep(0.2)  # give the OS a moment to actually switch focus

        # Verify it actually worked - SetForegroundWindow can return
        # without raising an exception yet still fail to actually change
        # which window is in the foreground.
        actual_foreground = win32gui.GetForegroundWindow()
        if actual_foreground != self.hwnd:
            print(f"WARNING: focus verification failed - foreground window "
                  f"is {actual_foreground}, expected {self.hwnd}. "
                  f"Inputs will likely NOT reach the game.")
            return False

        return True

    # -- low-level memory helpers -----------------------------------------

    def _read_int(self, offset):
        return self.pm.read_int(self.base + offset)

    def _resolve_pointer_chain(self, base_offset, offset_chain):
        """
        Mirrors the exact algorithm from the open-source 0xvpr/HM3-Trainer
        teleport hack (memory.hpp's find_dynamic_address): starting at
        exe_base + base_offset, for each offset in the chain, dereference
        the current address then add the offset. Returns the final
        resolved address, or None if a null pointer is hit along the way.
        """
        addr = self.base + base_offset
        for offset in offset_chain:
            try:
                addr = self.pm.read_uint(addr)
            except Exception:
                return None
            if addr == 0:
                return None
            addr += offset
        return addr

    def get_position(self):
        """
        Returns (x, y, z) player position, or None if the pointer chain
        couldn't be resolved (e.g. between missions, or during a menu).
        """
        addr = self._resolve_pointer_chain(PLAYER_XYZ_BASE, PLAYER_XYZ_OFFSETS)
        if addr is None:
            return None
        try:
            x = self.pm.read_float(addr)
            y = self.pm.read_float(addr + 4)
            z = self.pm.read_float(addr + 8)
            return (x, y, z)
        except Exception:
            return None

    def get_all_entity_positions(self):
        """
        Returns a list of (index, x, y, z) for every live entity (NPCs,
        targets, etc.) in the level - reverse-engineered from the
        open-source 0xvpr/HM3-Trainer (entity.hpp/hacks.cpp). Useful for
        identifying which index corresponds to a specific NPC (e.g. by
        standing near them and matching position).
        """
        try:
            entity_list_ptr = self.pm.read_uint(self.base + ENTITY_LIST_BASE)
            if entity_list_ptr == 0:
                return []
            n_entities = self.pm.read_uint(entity_list_ptr + 0x8)
            if n_entities < 7 or n_entities > 167:
                return []  # sanity check, same bounds the trainer uses

            results = []
            for i in range(n_entities):
                entity_ptr = self.pm.read_uint(entity_list_ptr + 0xC + i * 4)
                if entity_ptr == 0:
                    continue
                try:
                    x = self.pm.read_float(entity_ptr + ENTITY_XYZ_OFFSET)
                    y = self.pm.read_float(entity_ptr + ENTITY_XYZ_OFFSET + 4)
                    z = self.pm.read_float(entity_ptr + ENTITY_XYZ_OFFSET + 8)
                    results.append((i, x, y, z))
                except Exception:
                    continue
            return results
        except Exception:
            return []

    def get_state(self):
        """Snapshot of everything the fitness function needs."""
        return {
            "level": self._read_int(OFFSET_LEVEL),
            "time_ms": self._read_int(OFFSET_TIME),
            "witnesses": self._read_int(OFFSET_WITNESSES),
            "bodies_found": self._read_int(OFFSET_BODIES_FOUND),
            "covers_blown": self._read_int(OFFSET_COVERS_BLOWN),
            "targets_killed": self._read_int(OFFSET_TARGETS_KILLED),
            "position": self.get_position(),
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

    def release_all_inputs(self):
        """
        Force-release every key/button this bot ever presses, ignoring
        any errors (e.g. releasing something that wasn't actually down is
        harmless). This is the safety net against a crash mid-attempt
        leaving something stuck held down - which would break not just
        that attempt, but everything afterward (including restarting).
        """
        for key in ["w", "a", "s", "d", "e", "shift", "ctrl", "g"]:
            try:
                _scancode_key_up(key)
            except Exception:
                pass
        for button in ["left", "right"]:
            try:
                pydirectinput.mouseUp(button=button)
            except Exception:
                pass

    def key_down(self, key):
        _scancode_key_down(key)

    def key_up(self, key):
        _scancode_key_up(key)

    def mouse_down(self, button):
        pydirectinput.mouseDown(button=button)

    def mouse_up(self, button):
        pydirectinput.mouseUp(button=button)

    def scroll_now(self, clicks):
        if clicks != 0:
            _scroll_wheel(clicks)

    def send_key(self, key, duration=0.0):
        """Press and (optionally) hold a key for `duration` seconds."""
        _scancode_key_down(key)
        try:
            if duration > 0:
                time_module.sleep(duration)
        finally:
            _scancode_key_up(key)

    def send_mouse_move(self, dx, dy):
        pydirectinput.moveRel(dx, dy, relative=True)

    def send_click(self, button="left", duration=0.0):
        pydirectinput.mouseDown(button=button)
        try:
            if duration > 0:
                time_module.sleep(duration)
        finally:
            pydirectinput.mouseUp(button=button)

    def send_look(self, dx, dy, steps=6, step_delay=0.008):
        """
        Turn the camera during normal gameplay (not the weapon wheel).
        Split into several smaller sub-moves rather than one big
        instantaneous jump - a single large relative move can overshoot
        or "snap" in a way real continuous mouse input never would,
        which likely explains the jittery, inconsistent turning seen in
        genome playback.
        """
        if steps <= 1:
            pydirectinput.moveRel(dx, dy, relative=True)
            return

        # Distribute dx/dy across steps, carrying the rounding remainder
        # forward so the total moved still exactly matches dx/dy overall.
        remainder_x, remainder_y = 0.0, 0.0
        for _ in range(steps):
            step_x = dx / steps + remainder_x
            step_y = dy / steps + remainder_y
            move_x = round(step_x)
            move_y = round(step_y)
            remainder_x = step_x - move_x
            remainder_y = step_y - move_y
            pydirectinput.moveRel(move_x, move_y, relative=True)
            if step_delay > 0:
                time_module.sleep(step_delay)

    def send_toggle_holster(self):
        """A quick right-click tap (not a hold) holsters/unholsters whatever's currently equipped."""
        pydirectinput.mouseDown(button="right")
        try:
            time_module.sleep(0.05)
        finally:
            pydirectinput.mouseUp(button="right")

    def send_equip_weapon(self, scroll_clicks, hold_seconds=0.4):
        """
        Open the weapon wheel (hold right-click), scroll through the
        inventory by a number of clicks (positive/negative = opposite
        cycle directions), hold briefly so the selection registers, then
        release right-click to confirm whatever's currently selected.

        Wrapped in try/finally so the right mouse button always gets
        released even if something in between fails - otherwise a crash
        here leaves the button stuck down, breaking everything afterward
        (including restarting the mission).
        """
        pydirectinput.mouseDown(button="right")
        try:
            time_module.sleep(0.1)  # let the wheel actually open first
            if scroll_clicks != 0:
                _scroll_wheel(scroll_clicks)
            time_module.sleep(max(0.0, hold_seconds))
        finally:
            pydirectinput.mouseUp(button="right")

    # -- lifecycle -------------------------------------------------------

    def reset(self):
        """
        Restart the current mission attempt. Two different screens can be
        showing depending on how the last attempt ended:

          - Died: a screen with only 3 options (Restart, Load Game, Quit)
            appears automatically, Restart is already the default - just
            Enter confirms it.
          - Timed out (still alive): nothing is showing yet, need Escape
            to open the full pause menu (Resume Game, Options, Restart,
            Load Game, Save Game, Quit - Restart is 3rd), then navigate
            down to Restart.

        These two menus have different lengths, so the same blind
        navigation sequence can't safely handle both - applying the
        pause-menu navigation to the death screen would land on Quit and
        exit the game entirely. Instead: try the simple death-screen path
        first, and check whether anything in the game state actually
        changed compared to right before - not just whether it's now all
        zero, since a run can legitimately end already at all zero
        (nothing dirty happened), which would look identical to "nothing
        happened" if we only checked for zeros. Only fall back to the
        pause-menu navigation if truly nothing changed.
        """
        self.release_all_inputs()  # defensive: clear anything stuck from before
        self.focus_game_window()
        self._recenter_cursor()

        log = []  # buffer debug messages - printing mid-sequence risks the
                   # terminal stealing focus back from the game right when
                   # a critical keypress needs to land on the game instead.

        pre_state = self.get_state()
        log.append(f"[reset] pre_state: {pre_state}")

        # Try the simple path: assume a death screen with Restart already
        # selected, and Enter selects it - but that opens a "Restart?
        # Yes/No" confirmation popup, which still needs Left (move to
        # Yes) + Enter (confirm) before anything actually happens.
        _scancode_press("enter")
        time_module.sleep(1.5)  # real recordings show 0.85-1.4s here
        _scancode_press("left")
        time_module.sleep(1.0)  # real recordings show 0.7-0.85s here
        _scancode_press("enter")
        time_module.sleep(RESTART_SETTLE_SECONDS)
        post_state = self.get_state()
        log.append(f"[reset] post_state after Enter-only attempt: {post_state}")

        if post_state != pre_state:
            # Something changed - the death-screen path worked.
            log.append("[reset] took death-screen path (state changed)")
            print("\n".join(log))
            return post_state

        log.append("[reset] no change detected, falling back to pause-menu navigation")

        # Nothing changed at all - we were probably still alive (timed
        # out, not died), so nothing was showing yet and that Enter press
        # was a harmless no-op. Open the pause menu and navigate to
        # Restart explicitly instead.
        self.focus_game_window()
        _scancode_press("esc")
        time_module.sleep(0.4)  # real recording shows ~0.33-0.48s here

        # No "reset to top" navigation needed here - the pause menu always
        # opens fresh at Resume Game, confirmed by the real recording
        # (which never pressed Up at all). Just Down x2 to reach Restart.
        _scancode_press("down")
        time_module.sleep(0.15)
        _scancode_press("down")
        time_module.sleep(0.3)  # slightly longer before confirming selection, matches ~0.3s observed

        # Enter selects Restart, which opens the same Yes/No confirmation
        # popup - Left (move to Yes) + Enter (confirm) to actually go
        # through with it.
        _scancode_press("enter")
        time_module.sleep(1.5)  # real recordings show 0.85-1.4s here
        _scancode_press("left")
        time_module.sleep(1.0)  # real recordings show 0.7-0.85s here
        _scancode_press("enter")
        time_module.sleep(RESTART_SETTLE_SECONDS)

        state = self.get_state()
        if not self.is_clean(state) or state["time_ms"] != 0:
            # Give it a bit more time in case the transition was slow.
            time_module.sleep(1.0)
            state = self.get_state()

        log.append(f"[reset] pause-menu path final state: {state}")
        print("\n".join(log))
        return state

    def run_attempt(self, action_sequence):
        """
        Play back a full list of scheduled events, then wait for the
        mission to finish (or time out), and return the final state.

        action_sequence: list of (delay_seconds, callable) tuples, where
        each `callable` is a QUICK, NON-BLOCKING action - e.g. key_down,
        key_up, a mouse move, or a click. Concurrent inputs (like holding
        shift and W at the same time) work by scheduling a "down" event
        and a separate later "up" event, rather than one call that blocks
        for a whole duration - blocking would prevent anything else from
        happening at the same time.
        Delays are relative to the start of the attempt, not to each other.
        """
        # Re-focus right before actions actually start firing - focus may
        # have drifted back to the terminal during reset()'s menu
        # navigation delays, even though reset() already focused once.
        self.focus_game_window()

        start = time_module.time()
        idx = 0
        n = len(action_sequence)
        last_focus_check = start

        try:
            while True:
                now = time_module.time()
                elapsed = now - start

                # Periodically re-focus throughout the whole attempt, not
                # just once at the start - if focus gets stolen at any
                # point during a long run, every subsequent action would
                # silently go nowhere, and no scheduling/calibration fix
                # could compensate for that.
                if now - last_focus_check >= 2.0:
                    self.focus_game_window()
                    last_focus_check = now

                # Fire any actions whose scheduled time has arrived.
                while idx < n and action_sequence[idx][0] <= elapsed:
                    _, action_fn = action_sequence[idx]
                    action_fn()
                    idx += 1

                state = self.get_state()
                if self.is_finished(state):
                    return state

                if elapsed > MAX_ATTEMPT_SECONDS:
                    return state

                time_module.sleep(POLL_INTERVAL)
        finally:
            # No matter how this attempt ends (finished, timed out, or an
            # exception anywhere above), make sure nothing is left stuck
            # held down going into the next attempt.
            self.release_all_inputs()

            if elapsed > MAX_ATTEMPT_SECONDS:
                # Attempt got stuck / never finished - treat as a failure.
                return state

            time_module.sleep(POLL_INTERVAL)

    def score(self, state):
        """
        Lower is better. Four tiers, worst to best:

          4. Didn't finish, DIRTY   - already got spotted/left evidence.
             (broke SA)               Ranked only by targets killed - NO
                                       distance credit at all, so getting
                                       close never rewards a run that blew
                                       its cover to get there.
          3. Didn't finish, CLEAN   - still stealthy. Ranked by targets
             (still stealthy)         killed, then by distance to the
                                       vantage point as a tie-breaker -
                                       real gradient for "got closer while
                                       staying hidden."
          2. Finished, but broke   - better than not finishing, worse
             SA at some point        than any clean finish. Ranked by how
                                      minor the break was.
          1. Finished, clean SA    - ranked purely by time. Lower is
                                      faster.

        Tier gaps are large fixed constants so a run in a better tier
        always beats every run in a worse tier - critically, tier 3's
        worst possible score is still better than tier 4's best possible
        score, so distance can never outweigh staying undetected.
        """
        TIER_UNFINISHED_DIRTY = 3_000_000
        TIER_UNFINISHED_CLEAN = 2_000_000
        TIER_FINISHED_DIRTY = 1_000_000

        targets_killed = state["targets_killed"]

        if not self.is_finished(state):
            progress_credit = targets_killed * 100_000

            if not self.is_clean(state):
                # Already spotted/left evidence - no distance credit at
                # all, ranked only by targets killed so far.
                return TIER_UNFINISHED_DIRTY - progress_credit

            # Still stealthy - distance to the vantage point is a real,
            # continuous tie-breaker on top of targets killed. Clamped so
            # even a huge distance can never push this tier's score above
            # the dirty tier's boundary - the invariant (clean always
            # beats dirty) must hold no matter what distance is measured.
            distance = None
            if state.get("position") and VANTAGE_POINT:
                px, py, pz = state["position"]
                vx, vy, vz = VANTAGE_POINT
                distance = ((px - vx) ** 2 + (py - vy) ** 2 + (pz - vz) ** 2) ** 0.5
            distance_credit = min(distance, MAX_DISTANCE_CREDIT) if distance is not None else 0.0
            return TIER_UNFINISHED_CLEAN - progress_credit + distance_credit

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