import os
import sys
import glob

from parallel_screening import *

import sys
sys.path.append("/pine/scr/j/h/jhochuli/git/generative_pharmacophore")
from nsp13_scoring_function import get_nsp13_structural_reward

def score_mols(mols):

    scores = [get_nsp13_structural_reward(mol) for mol in mols]
    return scores

pattern = os.getenv('SLURM_ARRAY_TASK_ID')
filenames = list(glob.glob(f"/proj/tropshalab/shared/REAL_Library/plaintext/smile_all_{pattern}*.txt"))
#filenames = list(glob.glob(f"test_data/{pattern}*.txt"))
screener = UnifiedScreener(filenames, output_filename = f"nsp13_structural_hits_{pattern}.txt",
            mol_function = score_mols, log_filename = f"nsp13_log_{pattern}.txt",
            result_checker = lambda x: x > 0.9,
            delimiter = "", skip_first_line = True, smiles_position = 0)

screener.run()



