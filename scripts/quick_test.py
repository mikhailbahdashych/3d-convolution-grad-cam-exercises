"""Quick test script to verify everything works."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess

def run_command(cmd, description):
    """Run a command and print output."""
    print("\n" + "="*60)
    print(f"TEST: {description}")
    print("="*60)
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ {description} - PASSED")
    else:
        print(f"\n✗ {description} - FAILED")
        return False

    return True

def main():
    """Run quick tests."""
    print("\n" + "="*60)
    print("QUICK SYSTEM TEST")
    print("="*60)

    tests = [
        ("source .venv/bin/activate && python scripts/test_training_init.py",
         "Training initialization"),

        ("source .venv/bin/activate && python scripts/test_dataset.py",
         "Dataset loading"),
    ]

    results = []
    for cmd, desc in tests:
        result = run_command(cmd, desc)
        results.append((desc, result))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for desc, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{desc:.<50} {status}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYou're ready to train!")
        print("\nNext steps:")
        print("  1. Quick test (2 epochs):  python scripts/train.py --epochs 2 --batch-size 4")
        print("  2. Full training:          python scripts/train.py --epochs 50")
        print("  3. Monitor training:       tensorboard --logdir outputs/logs")
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("="*60)
        print("Please check the errors above.")

if __name__ == "__main__":
    main()
