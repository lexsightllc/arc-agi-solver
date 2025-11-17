# SPDX-License-Identifier: MPL-2.0
from setuptools import setup, find_packages
import os

# Safely read requirements
install_requires = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Safely read README
long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='arc-agi-solver',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'arc-solver = main:app',
        ],
    },
    author='ARC AGI Team',
    author_email='arc-agi@example.com',
    description='A neuro-symbolic program induction system for ARC AGI tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/arc-agi-solver',
    license='MPL-2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)
