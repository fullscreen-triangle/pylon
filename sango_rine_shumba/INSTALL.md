# Installation Guide - Sango Rine Shumba Network Validation Framework

## Quick Installation

### Method 1: Direct Installation (Recommended)
```bash
cd sango_rine_shumba
pip install -r requirements.txt
pip install -e .
```

### Method 2: Development Installation
```bash
cd sango_rine_shumba
pip install -r requirements-dev.txt
pip install -e ".[dev,docs,performance,research]"
```

### Method 3: Minimal Installation
```bash
cd sango_rine_shumba
pip install -e .  # Installs only core dependencies
```

## System Requirements

### Python Version
- **Required**: Python 3.8 or higher
- **Recommended**: Python 3.10+
- **Tested on**: Python 3.8, 3.9, 3.10, 3.11, 3.12

### Operating Systems
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+, Fedora 30+)
- ✅ **macOS** (10.15+ Catalina)
- ✅ **Windows** (Windows 10+)

### Hardware Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 1GB free space for framework + results
- **CPU**: Any modern x64 processor

## Dependency Information

### Core Dependencies (Always Required)
```
numpy>=1.21.0          # Mathematical operations
scipy>=1.7.0           # Scientific computing  
matplotlib>=3.5.0      # Plotting and visualization
seaborn>=0.11.0        # Statistical visualizations
pandas>=1.3.0          # Data manipulation
sympy>=1.9             # Symbolic mathematics
```

### Optional Dependencies

#### Development Tools
```
pytest>=6.0.0          # Testing framework
black>=21.0.0          # Code formatting
mypy>=0.910            # Type checking
flake8>=3.9.0          # Code linting
```

#### Performance Analysis
```
memory-profiler>=0.60.0  # Memory usage profiling
line-profiler>=3.3.0     # Line-by-line profiling
tqdm>=4.62.0             # Progress bars
```

#### Research Tools
```
jupyter>=1.0.0         # Interactive notebooks
networkx>=2.6.0        # Network analysis
scikit-learn>=1.0.0    # Machine learning
statsmodels>=0.12.0    # Statistical modeling
```

## Installation Verification

### Basic Verification
```bash
cd sango_rine_shumba/src
python -c "from observer.transcendent_observer import TranscendentObserver; print('✅ Observer framework OK')"
python -c "from network.ambigous_compressor import AmbiguousCompressor; print('✅ Network framework OK')"
```

### Full Validation Test
```bash
cd sango_rine_shumba/src
python simulation.py --run-observer --verbose
```

Expected output should show:
```
📊 Running observer framework validation...
✅ Core dependencies successfully installed
✅ Sango Rine Shumba modules successfully imported
✅ VALIDATION PASSED
```

## Common Installation Issues

### Issue 1: Python Version Error
**Error**: `RuntimeError: Sango Rine Shumba requires Python 3.8 or higher`
**Solution**: 
```bash
python --version  # Check current version
# Install Python 3.8+ from python.org
# Or use pyenv: pyenv install 3.10.0 && pyenv global 3.10.0
```

### Issue 2: NumPy Installation Fails  
**Error**: `Failed building wheel for numpy`
**Solution**:
```bash
# On Windows
pip install --upgrade setuptools wheel
pip install numpy

# On Linux/macOS
sudo apt-get install python3-dev  # Ubuntu
# or
brew install python@3.10          # macOS
```

### Issue 3: Matplotlib Backend Issues
**Error**: `No module named '_tkinter'`
**Solution**:
```bash
# Linux
sudo apt-get install python3-tk

# macOS  
brew install tcl-tk

# Or use non-interactive backend
export MPLBACKEND=Agg
```

### Issue 4: Import Errors During Validation
**Error**: `ModuleNotFoundError: No module named 'observer'`
**Solution**:
```bash
# Ensure you're in the src directory
cd sango_rine_shumba/src
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python simulation.py --run-all
```

### Issue 5: Permission Errors
**Error**: `PermissionError: [Errno 13] Permission denied`
**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Performance Optimization

### For Large-Scale Validation
```bash
# Increase memory limit for NumPy
export OMP_NUM_THREADS=4

# Use faster BLAS library (optional)
pip install intel-mkl

# For GPU acceleration (optional)
pip install cupy-cuda11x  # If CUDA 11.x available
```

### For Development
```bash
# Install development tools
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Enable fast imports
export PYTHONDONTWRITEBYTECODE=1
```

## Virtual Environment Setup (Recommended)

### Using venv
```bash
python -m venv sango-env
source sango-env/bin/activate  # Linux/macOS
# or
sango-env\Scripts\activate     # Windows

cd sango_rine_shumba
pip install -r requirements.txt
pip install -e .
```

### Using conda
```bash
conda create -n sango python=3.10
conda activate sango

cd sango_rine_shumba
pip install -r requirements.txt
pip install -e .
```

### Using pipenv
```bash
cd sango_rine_shumba
pipenv install -r requirements.txt
pipenv install -e .
pipenv shell
```

## IDE Configuration

### Visual Studio Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Add Local Interpreter → Existing Environment
3. Select `venv/bin/python`
4. Enable: Code → Reformat Code with Black

## Uninstallation

### Remove Package
```bash
pip uninstall sango-rine-shumba
```

### Clean Installation
```bash
# Remove all dependencies (be careful!)
pip freeze | xargs pip uninstall -y

# Or remove virtual environment
rm -rf venv/  # Linux/macOS
rmdir /s venv # Windows
```

## Next Steps

After successful installation:

1. **Read Documentation**: Check `README.md` for detailed usage
2. **Run Quick Test**: `python src/simulation.py --run-observer` 
3. **Explore Examples**: See code examples in README.md
4. **Full Validation**: `python src/simulation.py --run-all --verbose`
5. **Review Results**: Check `validation_results/` directory

## Getting Help

- **Documentation Issues**: Check README.md and source code comments
- **Installation Problems**: Review this guide and check system requirements
- **Runtime Errors**: Use `--verbose` flag for detailed logging
- **Performance Issues**: See Performance Optimization section above

---

**Ready to explore gear ratio-based hierarchical networks!** 🚀
