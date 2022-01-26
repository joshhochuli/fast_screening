Repository for running operations on RDKit molecules in parallel (QSAR models, pharmacophore screens, etc)

### Setup
1. Create a conda environment from the provided file: `conda env create -f environment.yml`
2. Activate the environment: `conda activate fast_screening`
3. Run tests (will spit out a lot of text, but shouldn't print errors): `python test.py TestAll`
4. Run template.py and check output file `example_output_filename.txt`: `python template.py`
5. Modify template.py to include desired files for screening, screening function, and output
