"""
Genetic algorithm for the Hitman: Blood Money - Curtains Down bot.

A "genome" is just a list of (start_time, key, duration) genes - simple,
serializable, easy to mutate/crossover - which gets translated into real
key presses only when actually playing it back through BloodMoneyEnv.

Run this while the game is running and loaded into the mission.
"""

import random
import sys
import time as time_module
import pydirectinput

from bloodmoney_env import BloodMoneyEnv

# ---------------------------------------------------------------------------
# Genome / gene configuration
# ---------------------------------------------------------------------------

# Keys the algorithm is allowed to use for plain movement/interact presses.
# "shift" included so genomes can discover sprinting (hold shift + a
# movement key at the same time) rather than only walking.
MOVEMENT_KEYS = ["w", "a", "s", "d", "e", "shift", "ctrl"]

# Gene types and their relative frequency when generating random genes.
# "move_key"     - plain press-and-hold of a movement/interact/crouch key
# "fire_click"   - left-click (fire), only useful once a weapon is equipped
# "equip_weapon" - hold right-click, aim at a wheel angle, release to equip
# "throw_item"   - hold G, aim (camera move) during the hold, release to throw
GENE_TYPE_WEIGHTS = {
    "move_key": 0.65,
    "snap_turn": 0.05,
    "fire_click": 0.08,
    "equip_weapon": 0.10,
    "throw_item": 0.08,
    "toggle_holster": 0.04,
}

# Calibrated from real testing: sending dx=300 via send_look produced a
# measured ~55-degree turn. This is the one number the whole snap-turn
# system is built on - everything else is derived from it.
UNITS_PER_DEGREE = 300 / 55  # ~5.45

# Separate calibration for the RECORDING side: two real west-to-north
# (90-degree) test turns captured 232 and 248 raw pynput units. This is
# a different scale than UNITS_PER_DEGREE above (recording and replay
# don't use the same units), needed to convert a recorded bucket's raw
# pixel sum into an actual degree estimate before snapping to the
# nearest discrete option.
RECORDING_UNITS_PER_DEGREE = 240 / 90  # ~2.67

# Discrete turn sizes the algorithm can choose from, in degrees. Each one
# translates to a precisely-calculated dx via UNITS_PER_DEGREE, so there's
# no scale ambiguity, no accumulated noise, and nothing to calibrate per-
# turn - unlike continuous look_turn, which needed multiplier/damping/
# smoothing fixes and still didn't produce reliable navigation.
SNAP_TURN_ANGLES = [15, 45, 90, 180]

# Minimum real-world angle (degrees) for a recorded turn to be considered
# intentional at all - below this, it's discarded as noise rather than
# mapped to the smallest snap option.
SNAP_TURN_MIN_ANGLE = 8

# How many scroll "clicks" to cycle through the inventory when equipping
# something. Keep modest - the wheel likely only has a handful of items,
# so a huge scroll range just wastes search space cycling past the end.

# throw_item's aim adjustment stays continuous (not a discrete snap-turn)
# since it's a single one-off nudge during a throw, not a repeated,
# compounding action like camera navigation was - much less prone to the
# drift/noise problems that made continuous look_turn unreliable.
MAX_AIM_DELTA = 150
AIM_SCALE_MULTIPLIER = 2.12  # same calibration as UNITS_PER_DEGREE is based on
MAX_SCROLL_CLICKS = 6

# How long to hold the wheel open (after aiming) before releasing, so the
# game has time to register the hover before confirming the selection.
WHEEL_HOLD_SECONDS = 0.4

MIN_GENOME_LENGTH = 8
MAX_GENOME_LENGTH = 550

MIN_GENE_DURATION = 0.05
MAX_GENE_DURATION = 1.5

# Genes are scheduled within this window (seconds into the attempt).
# Real recorded runs go out to ~65s, so this leaves headroom without
# letting randomly-inserted genes land past the MAX_ATTEMPT_SECONDS
# timeout further down.
MAX_START_TIME = 70.0

POPULATION_SIZE = 20
GENERATIONS = 50
ELITE_COUNT = 3          # top N carried over unchanged each generation
TOURNAMENT_SIZE = 4
MUTATION_RATE = 0.7      # probability each gene gets mutated (aggressive - overnight run plateaued for 19 straight generations, needs a lot more diversity to escape)
GENE_ADD_RATE = 0.30      # probability of inserting a new random gene
GENE_REMOVE_RATE = 0.30   # probability of removing a random gene

# Phase-based training: genes with start < this are frozen - never
# mutated, never removed, and no new genes get inserted before this time.
# This concentrates all mutation/crossover effort on the part of the run
# that's actually broken (e.g. the theater interior), instead of
# constantly re-risking a route to the door that already works reliably.
# Set to 0.0 to disable (train the whole genome normally).
#
# Raised from an earlier, too-low value: real navigation to the theater
# door takes well past the first 8 seconds, so anything shorter left
# most of the "get out of spawn and reach the objective" portion
# unprotected - meaning nearly every individual (all except the literal
# unmutated seed copy) could have that critical early navigation broken
# by mutation before ever being tested.
FREEZE_BEFORE_SECONDS = 0.0


def _weighted_gene_type():
    types = list(GENE_TYPE_WEIGHTS.keys())
    weights = list(GENE_TYPE_WEIGHTS.values())
    return random.choices(types, weights=weights, k=1)[0]


def random_gene():
    gene_type = _weighted_gene_type()
    start = round(random.uniform(0.0, MAX_START_TIME), 2)

    if gene_type == "move_key":
        return {
            "start": start,
            "type": "move_key",
            "key": random.choice(MOVEMENT_KEYS),
            "duration": round(random.uniform(MIN_GENE_DURATION, MAX_GENE_DURATION), 2),
        }
    elif gene_type == "fire_click":
        return {
            "start": start,
            "type": "fire_click",
            "duration": round(random.uniform(0.05, 0.3), 2),
        }
    elif gene_type == "snap_turn":
        return {
            "start": start,
            "type": "snap_turn",
            "direction": random.choice(["left", "right"]),
            "angle": random.choice(SNAP_TURN_ANGLES),
        }
    elif gene_type == "throw_item":
        return {
            "start": start,
            "type": "throw_item",
            "duration": round(random.uniform(0.2, 1.0), 2),
            "aim_dx": random.randint(-MAX_AIM_DELTA, MAX_AIM_DELTA),
            "aim_dy": random.randint(-MAX_AIM_DELTA // 2, MAX_AIM_DELTA // 2),
        }
    elif gene_type == "toggle_holster":
        return {"start": start, "type": "toggle_holster"}
    else:  # equip_weapon
        return {
            "start": start,
            "type": "equip_weapon",
            "scroll_clicks": random.randint(-MAX_SCROLL_CLICKS, MAX_SCROLL_CLICKS),
            "hold_seconds": round(random.uniform(0.3, 1.0), 2),
        }


def random_genome():
    length = random.randint(MIN_GENOME_LENGTH, MAX_GENOME_LENGTH)
    genome = [random_gene() for _ in range(length)]
    genome.sort(key=lambda g: g["start"])
    return genome


def genome_to_action_sequence(genome, env):
    """
    Translate a genome (list of gene dicts) into the (delay, callable) list
    that env.run_attempt() expects. move_key and fire_click each become
    TWO scheduled events (down, then up) rather than one blocking call, so
    that overlapping genes (e.g. holding shift and W at the same time to
    sprint) actually overlap instead of executing one after another.

    Also compensates for the weapon wheel pausing the game: each
    equip_weapon gene's hold_seconds is real wall-clock time our
    scheduler counts as "elapsed," but during which the game itself made
    zero progress (paused). Without compensation, every gene scheduled
    after an equip action fires out of sync with what's actually
    happening in-game, and this drift compounds with every subsequent
    equip action in the genome. We track cumulative pause time as we
    walk through genes in chronological order, and shift every
    subsequent gene's effective start time by that running total.
    """
    action_sequence = []
    pause_offset = 0.0

    for gene in sorted(genome, key=lambda g: g["start"]):
        gene_type = gene["type"]
        effective_start = gene["start"] + pause_offset

        if gene_type == "move_key":
            key = gene["key"]
            start = effective_start
            end = start + gene["duration"]
            action_sequence.append((start, lambda k=key: env.key_down(k)))
            action_sequence.append((end, lambda k=key: env.key_up(k)))

        elif gene_type == "fire_click":
            start = effective_start
            end = start + gene["duration"]
            action_sequence.append((start, lambda: pydirectinput.mouseDown(button="left")))
            action_sequence.append((end, lambda: pydirectinput.mouseUp(button="left")))

        elif gene_type == "snap_turn":
            angle = gene["angle"]
            direction = gene["direction"]
            dx = angle * UNITS_PER_DEGREE * (1 if direction == "right" else -1)
            action_sequence.append((effective_start, lambda dx_=dx: env.send_look(dx_, 0)))

        elif gene_type == "throw_item":
            start = effective_start
            end = start + gene["duration"]
            aim_dx, aim_dy = gene["aim_dx"], gene["aim_dy"]
            action_sequence.append((start, lambda: env.key_down("g")))
            # Aim partway through the hold, mirroring how it's actually done in-game.
            action_sequence.append((start + gene["duration"] * 0.5, lambda dx_=aim_dx, dy_=aim_dy: env.send_look(dx_, dy_)))
            action_sequence.append((end, lambda: env.key_up("g")))

        elif gene_type == "toggle_holster":
            start = effective_start
            action_sequence.append((start, lambda: env.mouse_down("right")))
            action_sequence.append((start + 0.05, lambda: env.mouse_up("right")))

        else:  # equip_weapon: hold right-click, scroll to cycle, release to confirm
            clicks = gene["scroll_clicks"]
            hold = gene["hold_seconds"]
            start = effective_start
            # Scheduled as 3 separate non-blocking events, not one blocking
            # call - a blocking call here would freeze run_attempt()'s
            # entire action queue for the whole hold duration (up to
            # ~2.2s in real recordings), badly distorting the timing of
            # everything else scheduled during that window.
            action_sequence.append((start, lambda: env.mouse_down("right")))
            action_sequence.append((start + 0.1, lambda c=clicks: env.scroll_now(c)))
            action_sequence.append((start + hold, lambda: env.mouse_up("right")))

            # The wheel pauses the game for the whole hold duration - all
            # subsequent genes need to shift later by this same amount to
            # stay in sync with actual in-game elapsed time.
            pause_offset += hold

    action_sequence.sort(key=lambda a: a[0])
    return action_sequence


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def mutate(genome):
    new_genome = []
    for gene in genome:
        gene = dict(gene)  # copy
        if gene["start"] < FREEZE_BEFORE_SECONDS:
            # Frozen - this part of the route already works, don't risk it.
            new_genome.append(gene)
            continue
        if random.random() < MUTATION_RATE:
            gene_type = gene["type"]

            if gene_type == "move_key":
                attr = random.choice(["start", "key", "duration"])
                if attr == "start":
                    gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "key":
                    gene["key"] = random.choice(MOVEMENT_KEYS)
                else:
                    gene["duration"] = max(
                        MIN_GENE_DURATION,
                        round(gene["duration"] + random.uniform(-0.3, 0.3), 2),
                    )

            elif gene_type == "fire_click":
                attr = random.choice(["start", "duration"])
                if attr == "start":
                    gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                else:
                    gene["duration"] = max(0.05, round(gene["duration"] + random.uniform(-0.1, 0.1), 2))

            elif gene_type == "snap_turn":
                attr = random.choice(["start", "direction", "angle"])
                if attr == "start":
                    gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "direction":
                    gene["direction"] = "left" if gene["direction"] == "right" else "right"
                else:
                    gene["angle"] = random.choice(SNAP_TURN_ANGLES)

            elif gene_type == "throw_item":
                attr = random.choice(["start", "duration", "aim_dx", "aim_dy"])
                if attr == "start":
                    gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "duration":
                    gene["duration"] = max(0.1, round(gene["duration"] + random.uniform(-0.2, 0.2), 2))
                elif attr == "aim_dx":
                    gene["aim_dx"] = max(-MAX_AIM_DELTA, min(MAX_AIM_DELTA, gene["aim_dx"] + random.randint(-20, 20)))
                else:
                    gene["aim_dy"] = max(-MAX_AIM_DELTA // 2, min(MAX_AIM_DELTA // 2, gene["aim_dy"] + random.randint(-10, 10)))

            elif gene_type == "toggle_holster":
                gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))

            else:  # equip_weapon
                attr = random.choice(["start", "scroll_clicks", "hold_seconds"])
                if attr == "start":
                    gene["start"] = max(FREEZE_BEFORE_SECONDS, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "scroll_clicks":
                    gene["scroll_clicks"] = max(
                        -MAX_SCROLL_CLICKS,
                        min(MAX_SCROLL_CLICKS, gene["scroll_clicks"] + random.randint(-1, 1)),
                    )
                else:
                    gene["hold_seconds"] = max(0.1, round(gene["hold_seconds"] + random.uniform(-0.15, 0.15), 2))

        new_genome.append(gene)

    if random.random() < GENE_ADD_RATE and len(new_genome) < MAX_GENOME_LENGTH:
        new_gene = random_gene()
        # Never insert a new gene into the frozen region.
        new_gene["start"] = max(new_gene["start"], FREEZE_BEFORE_SECONDS)
        new_genome.append(new_gene)

    # Only remove genes from the unfrozen region - never touch frozen ones.
    unfrozen_indices = [i for i, g in enumerate(new_genome) if g["start"] >= FREEZE_BEFORE_SECONDS]
    if random.random() < GENE_REMOVE_RATE and len(unfrozen_indices) > 0 and len(new_genome) > MIN_GENOME_LENGTH:
        new_genome.pop(random.choice(unfrozen_indices))

    new_genome.sort(key=lambda g: g["start"])
    return new_genome


def crossover(parent_a, parent_b):
    """
    Time-based crossover: pick a cut point in time, take genes before it
    from parent A and genes after it from parent B. Since genes are
    time-ordered, this keeps the offspring internally coherent instead of
    scrambling timing.
    """
    cut = random.uniform(0.0, MAX_START_TIME)
    child = [g for g in parent_a if g["start"] < cut] + [g for g in parent_b if g["start"] >= cut]
    child.sort(key=lambda g: g["start"])

    if len(child) < MIN_GENOME_LENGTH:
        child.extend(random_gene() for _ in range(MIN_GENOME_LENGTH - len(child)))
    elif len(child) > MAX_GENOME_LENGTH:
        child = child[:MAX_GENOME_LENGTH]

    return child


def _normalize_key(key):
    """
    Recorded key names can vary depending on what modifiers were held:
      - Shift held while pressing a letter reports the UPPERCASE letter
        (e.g. 'W' instead of 'w') - happens during sprinting.
      - Ctrl held while pressing a letter reports a raw ASCII control
        character instead of the letter (e.g. Ctrl+W -> '\\x17') - happens
        while crouch-moving.
      - Left/right variants of modifier keys report distinct names
        ('ctrl_l' vs a generic 'ctrl').
    Without normalizing these, most sprint/crouch movement gets silently
    missed since it won't match our plain lowercase key names.
    """
    if len(key) == 1 and 1 <= ord(key) <= 26:
        return chr(ord(key) + 96)  # '\x01' -> 'a', '\x17' -> 'w', etc.
    if key in ("ctrl_l", "ctrl_r"):
        return "ctrl"
    if key in ("alt_l", "alt_r"):
        return "alt"
    if key in ("shift_l", "shift_r"):
        return "shift"
    if len(key) == 1:
        return key.lower()
    return key


def _smooth_mouse_events(events, alpha=0.3):
    """
    Apply an exponential moving average to the stream of recorded mouse
    movements, in time order. Each smoothed value is a blend of the raw
    reading and the recent trend (alpha controls the blend - lower means
    more smoothing/inertia, higher stays closer to the raw signal).

    This suppresses quick back-and-forth hand jitter while preserving the
    actual underlying direction being turned toward - a more principled
    fix than bucketing raw values and then applying a blanket damping
    factor afterward, since it separates noise from signal at the source
    instead of just turning everything down uniformly.
    """
    smoothed = []
    ema_dx, ema_dy = 0.0, 0.0
    for ev in events:
        if ev["type"] == "mouse_move":
            ema_dx = alpha * ev["dx"] + (1 - alpha) * ema_dx
            ema_dy = alpha * ev["dy"] + (1 - alpha) * ema_dy
            new_ev = dict(ev)
            new_ev["dx"] = ema_dx
            new_ev["dy"] = ema_dy
            smoothed.append(new_ev)
        else:
            smoothed.append(ev)
    return smoothed


def recording_to_genome(events, look_bucket_seconds=0.4, smoothing_alpha=0.3):
    """
    Convert a raw recording (from record_run.py) into genome format.

    - key_down/key_up pairs for movement keys become move_key genes.
    - key_down/key_up pairs for 'g' become throw_item genes, with the aim
      computed from mouse movement that happened during the hold.
    - mouse_down/mouse_up "left" pairs become fire_click genes.
    - mouse_down/mouse_up "right" pairs (weapon wheel) become a single
      equip_weapon gene, with the angle computed from the net mouse
      movement that happened during the hold.
    - mouse_move events NOT inside a right-click or 'g' hold are first
      smoothed (EMA) to separate real turning intent from hand jitter,
      then bucketed into short time windows and summed, becoming
      look_turn genes.
    """
    import math

    events = _smooth_mouse_events(events, alpha=smoothing_alpha)

    genome = []
    open_keys = {}       # normalized key -> start_time
    right_hold_start = None
    right_hold_scroll = 0
    g_hold_start = None
    g_hold_dx = 0.0
    g_hold_dy = 0.0
    left_hold_start = None
    look_buckets = {}    # bucket_index -> [dx_sum, dy_sum]

    for ev in events:
        t = ev["time"]

        if ev["type"] == "key_down":
            key = _normalize_key(ev["key"])

            if key == "g":
                if g_hold_start is None:
                    g_hold_start = t
                    g_hold_dx = 0.0
                    g_hold_dy = 0.0
            elif key in MOVEMENT_KEYS and key not in open_keys:
                open_keys[key] = t

        elif ev["type"] == "key_up":
            key = _normalize_key(ev["key"])

            if key == "g":
                if g_hold_start is not None:
                    genome.append({
                        "start": round(g_hold_start, 2),
                        "type": "throw_item",
                        "duration": round(max(0.1, t - g_hold_start), 2),
                        "aim_dx": int(max(-MAX_AIM_DELTA, min(MAX_AIM_DELTA, g_hold_dx * AIM_SCALE_MULTIPLIER))),
                        "aim_dy": int(max(-MAX_AIM_DELTA // 2, min(MAX_AIM_DELTA // 2, g_hold_dy * AIM_SCALE_MULTIPLIER))),
                    })
                    g_hold_start = None
            elif key in open_keys:
                start = open_keys.pop(key)
                genome.append({
                    "start": round(start, 2),
                    "type": "move_key",
                    "key": key,
                    "duration": round(max(0.05, t - start), 2),
                })

        elif ev["type"] == "mouse_down" and ev["button"] == "left":
            left_hold_start = t

        elif ev["type"] == "mouse_up" and ev["button"] == "left":
            if left_hold_start is not None:
                genome.append({
                    "start": round(left_hold_start, 2),
                    "type": "fire_click",
                    "duration": round(max(0.05, t - left_hold_start), 2),
                })
                left_hold_start = None

        elif ev["type"] == "mouse_down" and ev["button"] == "right":
            right_hold_start = t
            right_hold_scroll = 0

        elif ev["type"] == "mouse_up" and ev["button"] == "right":
            if right_hold_start is not None:
                duration = t - right_hold_start
                if right_hold_scroll == 0 and duration < 0.2:
                    # Quick tap, no scroll - this is a holster/unholster
                    # toggle, not a real item switch.
                    genome.append({
                        "start": round(right_hold_start, 2),
                        "type": "toggle_holster",
                    })
                else:
                    genome.append({
                        "start": round(right_hold_start, 2),
                        "type": "equip_weapon",
                        "scroll_clicks": right_hold_scroll,
                        "hold_seconds": round(max(0.1, duration), 2),
                    })
                right_hold_start = None

        elif ev["type"] == "scroll":
            if right_hold_start is not None:
                right_hold_scroll += ev["dy"]

        elif ev["type"] == "mouse_move":
            if right_hold_start is not None:
                # The wheel is open (game paused) - mouse movement here
                # doesn't drive selection anymore (that's scroll now), so
                # just ignore it rather than feeding it into look_turn.
                pass
            elif g_hold_start is not None:
                # Accumulate toward the throw's aim instead of a separate
                # look_turn gene.
                g_hold_dx += ev["dx"]
                g_hold_dy += ev["dy"]
            else:
                bucket = int(t / look_bucket_seconds)
                if bucket not in look_buckets:
                    look_buckets[bucket] = [0, 0]
                look_buckets[bucket][0] += ev["dx"]
                look_buckets[bucket][1] += ev["dy"]

    for bucket, (dx, dy) in look_buckets.items():
        # Convert the bucket's raw recorded pixel sum into an actual
        # degree estimate, then snap to the nearest discrete option -
        # discard entirely if too small to be a deliberate turn at all.
        degrees = dx / RECORDING_UNITS_PER_DEGREE
        if abs(degrees) < SNAP_TURN_MIN_ANGLE:
            continue
        direction = "right" if degrees > 0 else "left"
        nearest_angle = min(SNAP_TURN_ANGLES, key=lambda a: abs(a - abs(degrees)))
        genome.append({
            "start": round(bucket * look_bucket_seconds, 2),
            "type": "snap_turn",
            "direction": direction,
            "angle": nearest_angle,
        })

    genome.sort(key=lambda g: g["start"])
    return genome


def tournament_select(population_with_scores):
    """Pick the best of a random handful - keeps some diversity vs. always picking the global best."""
    contenders = random.sample(population_with_scores, min(TOURNAMENT_SIZE, len(population_with_scores)))
    return min(contenders, key=lambda ps: ps[1])[0]  # lower score wins


# ---------------------------------------------------------------------------
# Main evolution loop
# ---------------------------------------------------------------------------

def evaluate(genome, env):
    env.reset()
    action_sequence = genome_to_action_sequence(genome, env)
    final_state = env.run_attempt(action_sequence)
    score = env.score(final_state)
    return score, final_state


def save_best_run(genome, score, final_state, generation, path="best_run.json"):
    """
    Save the best genome found so far to disk, with metadata - so
    progress persists even if the script crashes or gets interrupted,
    and so the result can be inspected or reused as a future seed.
    """
    import json
    import time as time_module_local

    data = {
        "score": score,
        "final_state": final_state,
        "generation": generation,
        "saved_at": time_module_local.strftime("%Y-%m-%d %H:%M:%S"),
        "genome": genome,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_evolution(seed_genomes=None):
    """
    seed_genomes: a list of genomes (e.g. from multiple real recordings)
    to seed the initial population with, or None for fully random.
    Each seed contributes an unmutated copy plus mutated variants, filling
    out the rest of the population - this way crossover has multiple
    genuinely different real strategies to combine from generation 1,
    rather than only ever refining a single recording via mutation.
    """
    env = BloodMoneyEnv()

    population = []
    if seed_genomes:
        # One unmutated copy of each seed first, so the real recordings
        # themselves are always represented in the population.
        for seed in seed_genomes:
            if len(population) < POPULATION_SIZE:
                population.append(list(seed))

        # Fill the rest with mutated variants, cycling through the seeds
        # so each one gets roughly equal representation.
        i = 0
        while len(population) < POPULATION_SIZE:
            seed = seed_genomes[i % len(seed_genomes)]
            population.append(mutate(seed))
            i += 1
    else:
        population = [random_genome() for _ in range(POPULATION_SIZE)]

    best_ever_genome = None
    best_ever_score = float("inf")
    best_ever_state = None
    best_ever_generation = 0

    for generation in range(GENERATIONS):
        print(f"\n=== Generation {generation + 1}/{GENERATIONS} ===")

        scored_population = []
        for i, genome in enumerate(population):
            score, final_state = evaluate(genome, env)
            scored_population.append((genome, score))
            print(f"  Individual {i+1}/{len(population)}: score={score:.1f} state={final_state}")

            if score < best_ever_score:
                best_ever_score = score
                best_ever_genome = genome
                best_ever_state = final_state
                best_ever_generation = generation + 1
                print(f"  ^ New best score: {score:.1f}")
                save_best_run(genome, score, final_state, generation + 1)
                print(f"    (saved to best_run.json)")

        scored_population.sort(key=lambda ps: ps[1])

        # Elitism: carry the best few forward unchanged.
        next_population = [genome for genome, _ in scored_population[:ELITE_COUNT]]

        # Fill the rest via tournament selection + crossover + mutation.
        while len(next_population) < POPULATION_SIZE:
            parent_a = tournament_select(scored_population)
            parent_b = tournament_select(scored_population)
            child = crossover(parent_a, parent_b)
            child = mutate(child)
            next_population.append(child)

        population = next_population

    print(f"\n=== Done. Best score: {best_ever_score:.1f} (generation {best_ever_generation}) ===")
    print(f"Best final state: {best_ever_state}")
    print(f"Best genome saved to best_run.json ({len(best_ever_genome) if best_ever_genome else 0} genes)")
    return best_ever_genome, best_ever_score


if __name__ == "__main__":
    import json

    seeds = []
    if len(sys.argv) > 1:
        for path in sys.argv[1:]:
            print(f"Loading {path}...")
            with open(path, "r") as f:
                data = json.load(f)

            if isinstance(data, dict) and "genome" in data:
                # This is a saved best_run.json  genome is already built.
                genome = data["genome"]
                print(f"  -> loaded saved genome ({len(genome)} genes, "
                      f"score={data.get('score')}, from generation {data.get('generation')})")
            else:
                genome = recording_to_genome(data)
                print(f"  -> converted recording to genome with {len(genome)} genes")

            seeds.append(genome)
        print()

    run_evolution(seed_genomes=seeds if seeds else None)