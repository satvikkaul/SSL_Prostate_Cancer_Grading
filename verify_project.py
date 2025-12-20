#!/usr/bin/env python3
"""
Quick verification script to check project structure before pushing.
"""

import os
import sys

def check_file_exists(filename):
    """Check if a file exists and return status."""
    exists = os.path.exists(filename)
    status = "✓" if exists else "✗"
    print(f"{status} {filename}")
    return exists

def main():
    """Verify all required files exist."""
    print("=" * 60)
    print("PROJECT VERIFICATION")
    print("=" * 60)
    
    # Check Python files
    print("\n📦 SimCLR Implementation:")
    python_files = [
        "simclr_model.py",
        "simclr_augmentations.py", 
        "simclr_pretrain.py",
        "fine_tune_simclr.py",
        "evaluate_simclr.py",
    ]
    py_ok = all(check_file_exists(f) for f in python_files)
    
    print("\n📦 Baseline Implementation:")
    baseline_files = [
        "train_baseline.py",
        "evaluate_baseline.py",
    ]
    bl_ok = all(check_file_exists(f) for f in baseline_files)
    
    print("\n📦 Comparison & Analysis:")
    comp_files = ["compare_all_models.py"]
    comp_ok = all(check_file_exists(f) for f in comp_files)
    
    print("\n📦 Existing Code (Autoencoder):")
    existing_files = [
        "Main.py",
        "fine_tune.py", 
        "evaluate_final.py",
        "variational_autoencoder.py",
        "my_data_generator.py",
        "setup_data.py",
    ]
    ex_ok = all(check_file_exists(f) for f in existing_files)
    
    print("\n📄 Documentation:")
    doc_files = [
        "README.md",
        "GPU_TRAINING_GUIDE.md",
        "COLAB_TRAINING_GUIDE.md",
        "PUSH_CHECKLIST.md",
        "Model_architecture.md",
    ]
    doc_ok = all(check_file_exists(f) for f in doc_files)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    all_ok = py_ok and bl_ok and comp_ok and ex_ok and doc_ok
    
    print(f"SimCLR files:      {'✓ OK' if py_ok else '✗ MISSING'}")
    print(f"Baseline files:    {'✓ OK' if bl_ok else '✗ MISSING'}")
    print(f"Comparison files:  {'✓ OK' if comp_ok else '✗ MISSING'}")
    print(f"Existing files:    {'✓ OK' if ex_ok else '✗ MISSING'}")
    print(f"Documentation:     {'✓ OK' if doc_ok else '✗ MISSING'}")
    
    # Count files
    new_files = python_files + baseline_files + comp_files
    print(f"\n📊 Statistics:")
    print(f"   New Python files: {len(new_files)}")
    print(f"   Documentation files: {len(doc_files)}")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ PROJECT READY TO PUSH!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Add SimCLR implementation and docs'")
        print("3. git push origin main")
        return 0
    else:
        print("❌ MISSING FILES - DO NOT PUSH YET")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
