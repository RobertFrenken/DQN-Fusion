#!/usr/bin/env python3
"""
Test script for Pydantic CLI validation.

Tests the design principles enforcement:
- PRINCIPLE 1: All folder structure parameters are explicit
- PRINCIPLE 8: Fail-early prerequisite checking
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cli.pydantic_validators import validate_cli_config, CANGraphCLIConfig


def test_valid_configs():
    """Test valid configurations."""
    print("="*70)
    print("Testing VALID configurations")
    print("="*70)

    test_cases = [
        {
            "name": "VGAE Autoencoder (Teacher)",
            "model": "vgae",
            "dataset": "hcrl_ch",
            "mode": "autoencoder",
            "learning_type": "unsupervised",
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "GAT Normal (Teacher)",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "normal",
            "learning_type": "supervised",
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "GAT Curriculum (Teacher)",
            "model": "gat",
            "dataset": "set_01",
            "mode": "curriculum",
            "learning_type": "supervised",
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "DQN Fusion (Teacher)",
            "model": "dqn",
            "dataset": "hcrl_sa",
            "mode": "fusion",
            "learning_type": "rl_fusion",
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "GAT Student with Distillation",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "distillation",
            "learning_type": "supervised",
            "distillation": "with-kd",
            "model_size": "student",
        },
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        name = test_case.pop("name")
        print(f"\n{'─'*70}")
        print(f"Test: {name}")
        print(f"{'─'*70}")

        try:
            config = validate_cli_config(**test_case, job_type="single")
            print(f"✅ PASS")
            print(f"    Save path: {config.canonical_save_path}")
            print(f"    Sample strategy: {config.sample_strategy}")
            passed += 1
        except ValueError as e:
            print(f"❌ FAIL: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Valid Config Tests: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")

    return failed == 0


def test_invalid_configs():
    """Test invalid configurations that should fail validation."""
    print("="*70)
    print("Testing INVALID configurations (should fail)")
    print("="*70)

    test_cases = [
        {
            "name": "VGAE with supervised learning (INVALID)",
            "model": "vgae",
            "dataset": "hcrl_ch",
            "mode": "normal",
            "learning_type": "supervised",  # ❌ VGAE requires unsupervised
            "distillation": "no-kd",
            "model_size": "teacher",
            "expected_error": "VGAE model requires learning_type='unsupervised'"
        },
        {
            "name": "DQN with supervised learning (INVALID)",
            "model": "dqn",
            "dataset": "hcrl_ch",
            "mode": "normal",
            "learning_type": "supervised",  # ❌ DQN requires rl_fusion
            "distillation": "no-kd",
            "model_size": "teacher",
            "expected_error": "DQN model requires learning_type='rl_fusion'"
        },
        {
            "name": "GAT with unsupervised learning (INVALID)",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "autoencoder",
            "learning_type": "unsupervised",  # ❌ GAT requires supervised
            "distillation": "no-kd",
            "model_size": "teacher",
            "expected_error": "GAT model requires learning_type='supervised'"
        },
        {
            "name": "Unsupervised with normal mode (INVALID)",
            "model": "vgae",
            "dataset": "hcrl_ch",
            "mode": "normal",  # ❌ Unsupervised requires autoencoder mode
            "learning_type": "unsupervised",
            "distillation": "no-kd",
            "model_size": "teacher",
            "expected_error": "Unsupervised learning requires mode='autoencoder'"
        },
        {
            "name": "Distillation without with-kd flag (INVALID)",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "distillation",
            "learning_type": "supervised",
            "distillation": "no-kd",  # ❌ Distillation mode requires with-kd
            "model_size": "student",
            "expected_error": "Distillation mode requires distillation='with-kd'"
        },
        {
            "name": "Knowledge distillation with teacher size (INVALID)",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "distillation",
            "learning_type": "supervised",
            "distillation": "with-kd",
            "model_size": "teacher",  # ❌ with-kd requires student
            "expected_error": "Knowledge distillation (with-kd) requires model_size='student'"
        },
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        name = test_case.pop("name")
        expected_error = test_case.pop("expected_error")
        print(f"\n{'─'*70}")
        print(f"Test: {name}")
        print(f"{'─'*70}")
        print(f"Expected error: {expected_error}")

        try:
            config = validate_cli_config(**test_case, job_type="single")
            print(f"❌ FAIL: Should have raised ValueError but didn't")
            failed += 1
        except ValueError as e:
            error_msg = str(e)
            if expected_error in error_msg:
                print(f"✅ PASS: Correctly rejected invalid config")
                print(f"    Error message: {error_msg.split(chr(10))[0]}...")
                passed += 1
            else:
                print(f"❌ FAIL: Wrong error message")
                print(f"    Expected: {expected_error}")
                print(f"    Got: {error_msg}")
                failed += 1

    print(f"\n{'='*70}")
    print(f"Invalid Config Tests: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")

    return failed == 0


def test_backward_compatibility():
    """Test backward compatibility with auto-computation."""
    print("="*70)
    print("Testing BACKWARD COMPATIBILITY (auto-computation)")
    print("="*70)

    test_cases = [
        {
            "name": "Auto-compute learning_type from autoencoder mode",
            "model": "vgae",
            "dataset": "hcrl_ch",
            "mode": "autoencoder",
            # learning_type NOT provided - should auto-compute to "unsupervised"
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "Auto-compute learning_type from fusion mode",
            "model": "dqn",
            "dataset": "hcrl_ch",
            "mode": "fusion",
            # learning_type NOT provided - should auto-compute to "rl_fusion"
            "distillation": "no-kd",
            "model_size": "teacher",
        },
        {
            "name": "Auto-compute distillation from distillation mode",
            "model": "gat",
            "dataset": "hcrl_ch",
            "mode": "distillation",
            "learning_type": "supervised",
            "model_size": "student",
            # distillation NOT provided - should auto-compute to "with-kd"
        },
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        name = test_case.pop("name")
        print(f"\n{'─'*70}")
        print(f"Test: {name}")
        print(f"{'─'*70}")

        try:
            config = validate_cli_config(**test_case, job_type="single")
            print(f"✅ PASS (with warning)")
            print(f"    Learning type: {config.learning_type}")
            print(f"    Distillation: {config.distillation}")
            passed += 1
        except ValueError as e:
            print(f"❌ FAIL: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Backward Compatibility Tests: {passed} passed, {failed} failed")
    print(f"{'='*70}\n")

    return failed == 0


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("PYDANTIC CLI VALIDATION TESTS")
    print("="*70 + "\n")

    results = []

    # Test 1: Valid configurations
    results.append(("Valid Configs", test_valid_configs()))

    # Test 2: Invalid configurations
    results.append(("Invalid Configs", test_invalid_configs()))

    # Test 3: Backward compatibility
    results.append(("Backward Compatibility", test_backward_compatibility()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {name}")

    all_passed = all(result[1] for result in results)
    print("="*70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!\n")
        print("Design principles enforced:")
        print("  ✅ PRINCIPLE 1: All folder structure parameters are explicit")
        print("  ✅ PRINCIPLE 8: P→Q validation rules enforced")
        print("  ✅ Backward compatibility maintained (with warnings)")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
