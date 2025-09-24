#!/usr/bin/env python3
"""
Setup script for Sango Rine Shumba Demo
Handles virtual environment creation and dependency installation
"""

import os
import sys
import subprocess
import venv
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False

def create_virtual_environment():
    """Create a fresh virtual environment"""
    demo_dir = Path(__file__).parent
    venv_dir = demo_dir / ".venv"
    
    # Remove existing venv if it exists
    if venv_dir.exists():
        print("🗑️  Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_dir)
    
    print("🏗️  Creating new virtual environment...")
    venv.create(venv_dir, with_pip=True, clear=True)
    
    # Determine the correct python and pip paths
    if sys.platform == "win32":
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    return python_exe, pip_exe

def install_dependencies(pip_exe, demo_dir):
    """Install dependencies using pip"""
    requirements_file = demo_dir / "requirements.txt"
    
    # Upgrade pip first
    if not run_command(f'"{pip_exe}" install --upgrade pip setuptools wheel', 
                      "Upgrading pip and setuptools"):
        return False
    
    # Install requirements
    if not run_command(f'"{pip_exe}" install -r "{requirements_file}"', 
                      "Installing dependencies"):
        return False
    
    return True

def verify_installation(python_exe):
    """Verify that key packages can be imported"""
    print("🔍 Verifying installation...")
    
    test_imports = [
        "import numpy",
        "import pandas", 
        "import matplotlib",
        "import plotly",
        "import dash",
        "import aiohttp",
        "import requests",
        "import asyncio"
    ]
    
    for import_statement in test_imports:
        try:
            result = subprocess.run([str(python_exe), "-c", import_statement], 
                                  check=True, capture_output=True)
            module_name = import_statement.split()[1]
            print(f"   ✅ {module_name}")
        except subprocess.CalledProcessError:
            module_name = import_statement.split()[1]
            print(f"   ❌ {module_name} failed to import")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Sango Rine Shumba Demo Environment")
    print("=" * 50)
    
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    # Step 1: Create virtual environment
    try:
        python_exe, pip_exe = create_virtual_environment()
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False
    
    # Step 2: Install dependencies
    if not install_dependencies(pip_exe, demo_dir):
        print("❌ Failed to install dependencies")
        return False
    
    # Step 3: Verify installation
    if not verify_installation(python_exe):
        print("❌ Installation verification failed")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate"
        python_cmd = ".venv\\Scripts\\python.exe"
    else:
        activate_cmd = "source .venv/bin/activate"
        python_cmd = ".venv/bin/python"
    
    print(f"1. Activate environment: {activate_cmd}")
    print(f"2. Run demo: {python_cmd} run_demo.py")
    print(f"3. Access dashboard at: http://localhost:8050")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
