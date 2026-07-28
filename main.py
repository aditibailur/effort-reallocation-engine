"""
main.py — runs the full Reallocation Engine pipeline end-to-end.
Usage: python3 main.py
(run from the repo root)
"""
import subprocess
import sys

STEPS = [
    ("Generating synthetic dataset", "src/generate_data.py"),
    ("Running GIGO gate", "src/gigo_gate.py"),
    ("Running core engine (allocation + uncertainty)", "src/engine.py"),
    ("Running bias audit", "src/bias_audit.py"),
    ("Running explainability + critique", "src/explainability.py"),
    ("Running causal analysis (Pearl's 3 rungs)", "src/causal_analysis.py"),
    ("Running adversarial robustness test", "src/adversarial_test.py"),
    ("Running delegation map + hard-stop gate demo", "src/delegation_gate.py"),
]

if __name__ == "__main__":
    for label, script in STEPS:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"FAILED at: {script}")
            sys.exit(1)
    print("\nAll steps completed. See reports/ for output CSVs and JSON.")
