#!/usr/bin/env python
"""
Cleanup memory and kill lingering Python processes.
Run this before training or detection if you encounter memory errors.

Usage:
    python scripts/cleanup_memory.py
"""

import gc
import sys
import subprocess
from pathlib import Path

def cleanup_memory():
    """Clear Python memory."""
    print("Cleaning up Python memory...")
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ Cleared CUDA cache")
    except ImportError:
        pass
    
    print("✓ Garbage collection complete")


def kill_python_processes():
    """Kill lingering Python processes (Windows only)."""
    if sys.platform != 'win32':
        print("Process killing only supported on Windows")
        return
    
    try:
        print("\nKilling lingering Python processes...")
        subprocess.run(
            ['powershell', '-Command', 
             'Stop-Process -Name python -Force -ErrorAction SilentlyContinue'],
            check=False
        )
        import time
        time.sleep(2)
        print("✓ Processes terminated")
    except Exception as e:
        print(f"Warning: Could not kill processes: {e}")


def main():
    print("=" * 60)
    print("MEMORY CLEANUP UTILITY")
    print("=" * 60)
    
    cleanup_memory()
    
    # Ask before killing processes (or use --force flag)
    if sys.platform == 'win32':
        if '--force' in sys.argv or '-f' in sys.argv:
            kill_python_processes()
        else:
            response = input("\nKill lingering Python processes? [y/N]: ").lower()
            if response == 'y':
                kill_python_processes()
    
    print("\n✓ Cleanup complete!")
    print("\nYou can now run training or detection commands.")
    print("\nTip: Use 'python scripts/cleanup_memory.py --force' to skip prompt")


if __name__ == '__main__':
    main()
