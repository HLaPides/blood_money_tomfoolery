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
    "move_key": 0.55,
    "look_turn": 0.15,
    "fire_click": 0.10,
    "equip_weapon": 0.10,
    "throw_item": 0.10,
}

# How far a single look_turn gene can rotate the camera, in raw mouse-move
# pixels. Kept modest since these can stack across a genome.
MAX_LOOK_DELTA = 80

# How many scroll "clicks" to cycle through the inventory when equipping
# something. Keep modest - the wheel likely only has a handful of items,
# so a huge scroll range just wastes search space cycling past the end.
MAX_SCROLL_CLICKS = 6

# How long to hold the wheel open (after aiming) before releasing, so the
# game has time to register the hover before confirming the selection.
WHEEL_HOLD_SECONDS = 0.4

MIN_GENOME_LENGTH = 8
MAX_GENOME_LENGTH = 300

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
MUTATION_RATE = 0.25      # probability each gene gets mutated
GENE_ADD_RATE = 0.1       # probability of inserting a new random gene
GENE_REMOVE_RATE = 0.1    # probability of removing a random gene


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
    elif gene_type == "look_turn":
        return {
            "start": start,
            "type": "look_turn",
            "dx": random.randint(-MAX_LOOK_DELTA, MAX_LOOK_DELTA),
            "dy": random.randint(-MAX_LOOK_DELTA // 2, MAX_LOOK_DELTA // 2),  # vertical look is usually more limited
        }
    elif gene_type == "throw_item":
        return {
            "start": start,
            "type": "throw_item",
            "duration": round(random.uniform(0.2, 1.0), 2),
            "aim_dx": random.randint(-MAX_LOOK_DELTA, MAX_LOOK_DELTA),
            "aim_dy": random.randint(-MAX_LOOK_DELTA // 2, MAX_LOOK_DELTA // 2),
        }
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
    """
    action_sequence = []
    for gene in genome:
        gene_type = gene["type"]

        if gene_type == "move_key":
            key = gene["key"]
            start = gene["start"]
            end = start + gene["duration"]
            action_sequence.append((start, lambda k=key: env.key_down(k)))
            action_sequence.append((end, lambda k=key: env.key_up(k)))

        elif gene_type == "fire_click":
            start = gene["start"]
            end = start + gene["duration"]
            action_sequence.append((start, lambda: pydirectinput.mouseDown(button="left")))
            action_sequence.append((end, lambda: pydirectinput.mouseUp(button="left")))

        elif gene_type == "look_turn":
            dx, dy = gene["dx"], gene["dy"]
            action_sequence.append((gene["start"], lambda dx_=dx, dy_=dy: env.send_look(dx_, dy_)))

        elif gene_type == "throw_item":
            start = gene["start"]
            end = start + gene["duration"]
            aim_dx, aim_dy = gene["aim_dx"], gene["aim_dy"]
            action_sequence.append((start, lambda: env.key_down("g")))
            # Aim partway through the hold, mirroring how it's actually done in-game.
            action_sequence.append((start + gene["duration"] * 0.5, lambda dx_=aim_dx, dy_=aim_dy: env.send_look(dx_, dy_)))
            action_sequence.append((end, lambda: env.key_up("g")))

        else:  # equip_weapon: hold right-click, scroll to cycle, release to confirm
            clicks = gene["scroll_clicks"]
            hold = gene["hold_seconds"]
            action_sequence.append((gene["start"], lambda c=clicks, h=hold: env.send_equip_weapon(c, hold_seconds=h)))

    action_sequence.sort(key=lambda a: a[0])
    return action_sequence


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def mutate(genome):
    new_genome = []
    for gene in genome:
        gene = dict(gene)  # copy
        if random.random() < MUTATION_RATE:
            gene_type = gene["type"]

            if gene_type == "move_key":
                attr = random.choice(["start", "key", "duration"])
                if attr == "start":
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
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
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                else:
                    gene["duration"] = max(0.05, round(gene["duration"] + random.uniform(-0.1, 0.1), 2))

            elif gene_type == "look_turn":
                attr = random.choice(["start", "dx", "dy"])
                if attr == "start":
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "dx":
                    gene["dx"] = max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA, gene["dx"] + random.randint(-20, 20)))
                else:
                    gene["dy"] = max(-MAX_LOOK_DELTA // 2, min(MAX_LOOK_DELTA // 2, gene["dy"] + random.randint(-10, 10)))

            elif gene_type == "throw_item":
                attr = random.choice(["start", "duration", "aim_dx", "aim_dy"])
                if attr == "start":
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "duration":
                    gene["duration"] = max(0.1, round(gene["duration"] + random.uniform(-0.2, 0.2), 2))
                elif attr == "aim_dx":
                    gene["aim_dx"] = max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA, gene["aim_dx"] + random.randint(-20, 20)))
                else:
                    gene["aim_dy"] = max(-MAX_LOOK_DELTA // 2, min(MAX_LOOK_DELTA // 2, gene["aim_dy"] + random.randint(-10, 10)))

            else:  # equip_weapon
                attr = random.choice(["start", "scroll_clicks", "hold_seconds"])
                if attr == "start":
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                elif attr == "scroll_clicks":
                    gene["scroll_clicks"] = max(
                        -MAX_SCROLL_CLICKS,
                        min(MAX_SCROLL_CLICKS, gene["scroll_clicks"] + random.randint(-1, 1)),
                    )
                else:
                    gene["hold_seconds"] = max(0.1, round(gene["hold_seconds"] + random.uniform(-0.15, 0.15), 2))

        new_genome.append(gene)

    if random.random() < GENE_ADD_RATE and len(new_genome) < MAX_GENOME_LENGTH:
        new_genome.append(random_gene())

    if random.random() < GENE_REMOVE_RATE and len(new_genome) > MIN_GENOME_LENGTH:
        new_genome.pop(random.randrange(len(new_genome)))

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


def recording_to_genome(events, look_bucket_seconds=0.2):
    """
    Convert a raw recording (from record_run.py) into genome format.

    - key_down/key_up pairs for movement keys become move_key genes.
    - key_down/key_up pairs for 'g' become throw_item genes, with the aim
      computed from mouse movement that happened during the hold.
    - mouse_down/mouse_up "left" pairs become fire_click genes.
    - mouse_down/mouse_up "right" pairs (weapon wheel) become a single
      equip_weapon gene, with the angle computed from the net mouse
      movement that happened during the hold.
    - mouse_move events NOT inside a right-click or 'g' hold get bucketed
      into short time windows and summed, becoming look_turn genes -
      without this, a real recording produces hundreds of tiny, near-
      useless individual genes.
    """
    import math

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
                        "aim_dx": int(max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA, g_hold_dx))),
                        "aim_dy": int(max(-MAX_LOOK_DELTA // 2, min(MAX_LOOK_DELTA // 2, g_hold_dy))),
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
                genome.append({
                    "start": round(right_hold_start, 2),
                    "type": "equip_weapon",
                    "scroll_clicks": right_hold_scroll,
                    "hold_seconds": round(max(0.1, t - right_hold_start), 2),
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
        if dx == 0 and dy == 0:
            continue
        genome.append({
            "start": round(bucket * look_bucket_seconds, 2),
            "type": "look_turn",
            "dx": int(max(-MAX_LOOK_DELTA, min(MAX_LOOK_DELTA, dx))),
            "dy": int(max(-MAX_LOOK_DELTA // 2, min(MAX_LOOK_DELTA // 2, dy))),
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


def run_evolution(seed_genome=None):
    env = BloodMoneyEnv()

    # Seed the population with mutated copies of a known-good genome if
    # provided (e.g. an encoded version of your own PB run), otherwise
    # start fully random.
    population = []
    if seed_genome:
        population.append(list(seed_genome))
        while len(population) < POPULATION_SIZE:
            population.append(mutate(seed_genome))
    else:
        population = [random_genome() for _ in range(POPULATION_SIZE)]

    best_ever_genome = None
    best_ever_score = float("inf")

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
                print(f"  ^ New best score: {score:.1f}")

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

    print(f"\n=== Done. Best score: {best_ever_score:.1f} ===")
    print(f"Best genome: {best_ever_genome}")
    return best_ever_genome, best_ever_score


if __name__ == "__main__":
    import json

    seed = None
    if len(sys.argv) > 1:
        recording_path = sys.argv[1]
        print(f"Loading recording from {recording_path}...")
        with open(recording_path, "r") as f:
            recorded_events = json.load(f)
        seed = recording_to_genome(recorded_events)
        print(f"Converted to seed genome with {len(seed)} genes:")
        for gene in seed:
            print(f"  {gene}")
        print()

    run_evolution(seed_genome=seed)