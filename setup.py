from setuptools import setup, find_packages

setup(
    name='arc-agi-solver',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'arc-solver = main:app',
        ],
    },
    author='ARC AGI Team',
    author_email='arc-agi@example.com',
    description='A neuro-symbolic program induction system for ARC AGI tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/arc-agi-solver',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)
