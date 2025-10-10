#!/usr/bin/env python3
"""
Setup configuration for Sango Rine Shumba Network Validation Framework

This setup script configures the package for installation and distribution.
The framework implements gear ratio-based hierarchical communication with
transcendent observer architectures.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    raise RuntimeError("Sango Rine Shumba requires Python 3.8 or higher")

# Get the project root directory
here = Path(__file__).parent.resolve()

# Read the README file for long description
long_description = ""
readme_path = here / "README.md"
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from file, filtering out comments and built-in modules"""
    requirements = []
    req_file = here / filename
    
    if req_file.exists():
        with open(req_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and built-in module references
                if (line and 
                    not line.startswith('#') and 
                    not line.startswith('# ') and
                    '# Built into Python' not in line):
                    requirements.append(line)
    
    return requirements

# Core requirements (required for basic functionality)
install_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "pandas>=1.3.0",
    "sympy>=1.9",
]

# Development requirements (optional, for development and testing)
dev_requires = [
    "pytest>=6.0.0",
    "pytest-asyncio>=0.15.0", 
    "pytest-cov>=2.12.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "mypy>=0.910",
]

# Documentation requirements (optional, for building documentation)
docs_requires = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
]

# Performance analysis requirements (optional)
performance_requires = [
    "memory-profiler>=0.60.0",
    "line-profiler>=3.3.0",
    "tqdm>=4.62.0",
]

# Research and analysis requirements (optional)
research_requires = [
    "jupyter>=1.0.0",
    "ipython>=7.0.0", 
    "networkx>=2.6.0",
    "scikit-learn>=1.0.0",
    "statsmodels>=0.12.0",
    "pyyaml>=5.4.0",
]

# All optional requirements combined
all_requires = (dev_requires + docs_requires + 
               performance_requires + research_requires)

# Package metadata
setup(
    # Basic package information
    name="sango-rine-shumba",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="Gear Ratio-Based Hierarchical Network with Transcendent Observer Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # URLs and project information
    url="https://github.com/example/sango-rine-shumba",
    project_urls={
        "Bug Reports": "https://github.com/example/sango-rine-shumba/issues",
        "Source": "https://github.com/example/sango-rine-shumba",
        "Documentation": "https://sango-rine-shumba.readthedocs.io/",
    },
    
    # Package discovery and structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml", "*.json"],
    },
    
    # Python and dependency requirements
    python_requires=">=3.8",
    install_requires=install_requires,
    
    # Optional dependency groups
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires, 
        "performance": performance_requires,
        "research": research_requires,
        "all": all_requires,
    },
    
    # Entry points for command-line interfaces
    entry_points={
        "console_scripts": [
            "sango-validate=simulation:main",
            "sango-observer=observer.transcendent_observer:main",
            "sango-compressor=network.ambigous_compressor:main",
        ],
    },
    
    # Package classification
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        
        # Topic classification
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics", 
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Networking",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Operating system
        "Operating System :: OS Independent",
        
        # Environment
        "Environment :: Console",
        "Environment :: Other Environment",
    ],
    
    # Keywords for package discovery
    keywords=[
        "gear-ratios",
        "hierarchical-networks", 
        "transcendent-observer",
        "oscillatory-hierarchy",
        "compression-resistance",
        "bit-rot-protection",
        "network-validation",
        "mathematical-networks",
        "observer-pattern",
        "semantic-compression",
    ],
    
    # Minimum requirements check
    zip_safe=False,
    
    # Test suite configuration
    test_suite="tests",
    tests_require=dev_requires,
    
    # Additional metadata
    platforms=["any"],
)

# Post-installation validation
def validate_installation():
    """Validate that the package was installed correctly"""
    try:
        import numpy
        import matplotlib
        import scipy
        print("✅ Core dependencies successfully installed")
        
        # Try importing main modules
        from observer.transcendent_observer import TranscendentObserver
        from network.ambigous_compressor import AmbiguousCompressor
        print("✅ Sango Rine Shumba modules successfully imported")
        
        print("🎉 Installation validation completed successfully!")
        print("📚 See README.md for usage instructions")
        print("🚀 Run 'python src/simulation.py --run-all' to start validation")
        
    except ImportError as e:
        print(f"❌ Installation validation failed: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    # Run post-installation validation if setup was called directly
    if len(sys.argv) > 1 and sys.argv[1] in ["install", "develop"]:
        print("\n" + "="*60)
        print("SANGO RINE SHUMBA NETWORK VALIDATION FRAMEWORK")
        print("="*60)
        print("🔬 Research Framework for Gear Ratio-Based Networks")
        print("👁️  Transcendent Observer Architecture Implementation")
        print("🌊 8-Scale Oscillatory Hierarchy (10^-5 to 10^13 Hz)")
        print("🔄 O(1) Hierarchical Navigation with Bit-Rot Resistance")
        print("="*60)
        
        # Note: validate_installation() would be called after actual installation
        # For setup.py, we just show the information