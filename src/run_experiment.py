#!/usr/bin/env python3
"""
One-Click Experiment Runner
===========================
This script automates the complete experiment workflow for cervical cancer classification.

Usage:
    python -m src.run_experiment

Workflow:
    1. Check and install dependencies
    2. Verify dataset availability
    3. Preprocess data
    4. Train models
    5. Generate report
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

WORKSPACE = Path(".")
DATA_DIR = WORKSPACE / "data"
RAW_DIR = DATA_DIR / "raw"


def run_command(cmd, cwd=None):
    """Run a shell command and capture output"""
    print(f"\n[RUN] {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        cwd=cwd,
        capture_output=True, 
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("[ERROR]", result.stderr)
    if result.returncode != 0:
        print(f"[FAILED] Command failed with exit code: {result.returncode}")
        sys.exit(1)
    return result


def check_dependencies():
    """Check and install dependencies"""
    print("=" * 70)
    print("Checking Dependencies")
    print("=" * 70)
    
    # Check Python version
    print(f"Python version: {platform.python_version()}")
    
    # Install dependencies
    requirements_file = WORKSPACE / "requirements.txt"
    if requirements_file.exists():
        print("Installing dependencies...")
        run_command(f"python3 -m pip install -r {requirements_file}")
    else:
        print("requirements.txt not found, installing basic dependencies...")
        run_command("python3 -m pip install torch torchvision numpy pandas scikit-learn scipy matplotlib seaborn Pillow joblib tqdm")


def verify_dataset():
    """Verify dataset availability"""
    print("\n" + "=" * 70)
    print("Verifying Dataset Availability")
    print("=" * 70)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset already exists
    expected_classes = ["superficial-intermediate", "parabasal", "koilocytes", "dyskeratotic", "metaplastic"]
    existing_classes = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
    
    if all(cls in existing_classes for cls in expected_classes):
        print("[OK] Dataset is available")
        return
    
    # Dataset not found, show manual download instructions
    print("[ERROR] Dataset not found")
    print("\n[INFO] Please download the SIPaKMeD dataset manually:")
    print("1. Visit: http://www.cs.uoi.gr/~marina/sipakmed.html")
    print("2. Fill out the request form")
    print("3. Download and extract to:", RAW_DIR)
    print("4. Ensure the directory structure:")
    print("   data/raw/")
    print("   ├── superficial-intermediate/")
    print("   ├── parabasal/")
    print("   ├── koilocytes/")
    print("   ├── dyskeratotic/")
    print("   └── metaplastic/")
    
    # Wait for user input
    input("\nPress Enter after downloading the dataset...")


def preprocess_data():
    """Preprocess data"""
    print("\n" + "=" * 70)
    print("Preprocessing Data")
    print("=" * 70)
    
    run_command("python -m src.preprocess")


def train_models():
    """Train models"""
    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)
    
    run_command("python -m src.train")


def generate_report():
    """Generate summary report"""
    print("\n" + "=" * 70)
    print("Generating Report")
    print("=" * 70)
    
    results_dir = WORKSPACE / "results" / "experiment_results"
    if results_dir.exists():
        report_file = results_dir / "REPORT.md"
        if report_file.exists():
            print(f"[OK] Report generated at: {report_file}")
            print("\nTo view the report:")
            print(f"  cat {report_file}")
        else:
            print("[ERROR] Report file not found")
    else:
        print("[ERROR] Results directory not found")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("One-Click Experiment Runner")
    print("Cervical Cancer Classification with SE Attention")
    print("=" * 70)
    
    try:
        check_dependencies()
        verify_dataset()
        preprocess_data()
        train_models()
        generate_report()
        
        print("\n" + "=" * 70)
        print("🎉 Experiment completed successfully!")
        print("Results saved to: results/experiment_results/")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n[INFO] Experiment interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()