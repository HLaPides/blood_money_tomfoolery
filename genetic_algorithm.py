"""
Genetic algorithm for the Hitman: Blood Money - Curtains Down bot.

A "genome" is just a list of (start_time, key, duration) genes - simple,
serializable, easy to mutate/crossover - which gets translated into real
key presses only when actually playing it back through BloodMoneyEnv.

Run this while the game is running and loaded into the mission.
"""

import random
import time as time_module

from bloodmoney_env import BloodMoneyEnv

# ---------------------------------------------------------------------------
# Genome / gene configuration
# ---------------------------------------------------------------------------

# Keys the algorithm is allowed to use for plain movement/interact presses.
MOVEMENT_KEYS = ["w", "a", "s", "d", "e"]

# Gene types and their relative frequency when generating random genes.
# "move_key"     - plain press-and-hold of a movement/interact key
# "fire_click"   - left-click (fire), only useful once a weapon is equipped
# "equip_weapon" - hold right-click, aim at a wheel angle, release to equip
GENE_TYPE_WEIGHTS = {
    "move_key": 0.65,
    "fire_click": 0.15,
    "equip_weapon": 0.20,
}

# How far (in mouse-move pixels) to move while the weapon wheel is open,
# to reach a given angle. Tune this against your actual sensitivity/wheel
# layout - it doesn't need to be exact, just consistent enough that the
# same angle reliably lands on the same slot.
WHEEL_RADIUS = 150

# How long to hold the wheel open (after aiming) before releasing, so the
# game has time to register the hover before confirming the selection.
WHEEL_HOLD_SECONDS = 0.4

MIN_GENOME_LENGTH = 8
MAX_GENOME_LENGTH = 25

MIN_GENE_DURATION = 0.05
MAX_GENE_DURATION = 1.5

# Genes are scheduled within this window (seconds into the attempt).
MAX_START_TIME = 60.0

POPULATION_SIZE = 4
GENERATIONS = 2
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
    else:  # equip_weapon
        return {
            "start": start,
            "type": "equip_weapon",
            "angle": round(random.uniform(0.0, 359.9), 1),
        }


def random_genome():
    length = random.randint(MIN_GENOME_LENGTH, MAX_GENOME_LENGTH)
    genome = [random_gene() for _ in range(length)]
    genome.sort(key=lambda g: g["start"])
    return genome


def genome_to_action_sequence(genome, env):
    """
    Translate a genome (list of gene dicts) into the (delay, callable) list
    that env.run_attempt() expects.
    """
    action_sequence = []
    for gene in genome:
        gene_type = gene["type"]

        if gene_type == "move_key":
            key = gene["key"]
            duration = gene["duration"]
            action_fn = lambda k=key, d=duration: env.send_key(k, duration=d)

        elif gene_type == "fire_click":
            duration = gene["duration"]
            action_fn = lambda d=duration: env.send_click("left", duration=d)

        else:  # equip_weapon: hold right-click, aim at the wheel angle, release
            angle = gene["angle"]
            action_fn = lambda a=angle: env.send_equip_weapon(a)

        action_sequence.append((gene["start"], action_fn))

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

            else:  # equip_weapon
                attr = random.choice(["start", "angle"])
                if attr == "start":
                    gene["start"] = max(0.0, round(gene["start"] + random.uniform(-1.0, 1.0), 2))
                else:
                    gene["angle"] = round((gene["angle"] + random.uniform(-30.0, 30.0)) % 360.0, 1)

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
    run_evolution()