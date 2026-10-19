"""Regenerate everything for the launch-audio core with one command:

    python tools/audio/build_all.py            # full: references -> profile -> assets -> profile -> renders/figures -> tests
    python tools/audio/build_all.py --offline  # skip fetching/analysing the NASA references (uses the committed analysis JSON)

Requirements: Python 3.11+, numpy, scipy, matplotlib, pytest; ffmpeg with libopus on PATH.
Deterministic: all synthesis seeds are fixed (synth_layers.SEEDS); renders depend only on the profile, the assets and
the Python reference trajectory. Runtime: ~6-10 min on a desktop CPU; downloads ~0.5 GB of MP3 on the first run
(cached in tools/audio/_cache, git-ignored).
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)


def step(name, fn):
    t = time.time()
    print(f"== {name}")
    fn()
    print(f"   done in {time.time() - t:.0f} s")


def main():
    offline = "--offline" in sys.argv
    import make_profile
    if not offline:
        import references
        step("references: fetch + analyse", lambda: references.analyse(True))
    step("profile (pass 1)", make_profile.main)
    import synth_layers
    step("synthesise assets", synth_layers.main)
    step("profile (pass 2, with asset calibration)", make_profile.main)
    import validate
    step("render + validate", validate.main)
    step("tests", lambda: subprocess.run([sys.executable, "-m", "pytest", os.path.join(HERE, "tests"), "-q"], check=True, cwd=ROOT))


if __name__ == "__main__":
    main()
