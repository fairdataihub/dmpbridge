"""Rotation experiments have been removed from this pipeline.

All samples are now used for evaluation directly.
Run experiments with:

    dmpbridge-experiment experiments/llama3.1-8b-wholedoc.yaml
    dmpbridge-experiment experiments/wholedoc.yaml
"""
import sys

print("Rotation experiments have been removed.")
print("Run:  dmpbridge-experiment experiments/llama3.1-8b-wholedoc.yaml")
sys.exit(0)
