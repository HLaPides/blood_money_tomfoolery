"""
Isolated test: replay the seed genome (from a real recording) exactly
once, by itself - no genetic algorithm, no mutation, no population.

This is to check whether genome playback faithfully reproduces the real
route at all. Watch closely: does 47 face/move the same direction as your
real recording from the very start, or does it diverge immediately, or
does it start fine and drift off gradually?

Usage:
    python replay_seed.py my_run.json
"""

import sys
import json
from bloodmoney_env import BloodMoneyEnv
from genetic_algorithm import recording_to_genome, genome_to_action_sequence

if len(sys.argv) < 2:
    print("Usage: python replay_seed.py <recording.json>")
    sys.exit(1)

with open(sys.argv[1], "r") as f:
    events = json.load(f)

genome = recording_to_genome(events)
print(f"Loaded genome with {len(genome)} genes.")

env = BloodMoneyEnv()
print("Resetting mission...")
env.reset()

print("Playing back genome now - watch closely from the very start...")
action_sequence = genome_to_action_sequence(genome, env)
final_state = env.run_attempt(action_sequence)

print(f"\nFinal state: {final_state}")