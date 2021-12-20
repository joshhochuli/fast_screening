<<<<<<< HEAD
import os
import sys
import glob
from rdkit.Chem.Scaffolds import MurckoScaffold

from parallel_screening import *

import sys
sys.path.append("/pine/scr/j/h/jhochuli/git/generative_pharmacophore")
from nsp13_scoring_function import get_nsp13_structural_reward

def score_mols(mols):

    scores = [get_nsp13_structural_reward(mol) for mol in mols]
    return scores

def longleaf_main():

  pattern = os.getenv('SLURM_ARRAY_TASK_ID')
  filenames = list(glob.glob(f"/proj/tropshalab/shared/REAL_Library/plaintext/smile_all_{pattern}*.txt"))
  #filenames = list(glob.glob(f"test_data/{pattern}*.txt"))
  screener = UnifiedScreener(filenames, output_filename = f"nsp13_structural_hits_{pattern}.txt",
              mol_function = score_mols, log_filename = f"nsp13_log_{pattern}.txt",
              result_checker = lambda x: x > 0.9,
              delimiter = "", skip_first_line = True, smiles_position = 0)

  screener.run()

def scaffold_write_function(filename, result_queue, result_checker, log_filename):

    def result_write(file_obj, result_dict):

        file_obj.write("start batch\n")
        for key, value in result_dict.items():
            file_obj.write(f"{key}, {value}\n")
        file_obj.write("end batch\n")

    f = open(filename, 'w')

    if log_filename == None:
        log_filename = "log.txt"
    log_f = open(log_filename, 'w')

    scaffolds = {}

    global_start_time = time.time()
    start_time = time.time()
    counter = 0
    batch_counter = 0
    while True:
        #result = result_queue.get(block = True, timeout = 1000)
        result = result_queue.get()
        if result == "EMPTY":

            result_write(f, scaffolds)
            f.close()
            log_f.close()
            return

        for x in result:
            murcko_scaffold = x[2]
            if murcko_scaffold not in scaffolds:
                scaffolds[murcko_scaffold] = 0
            scaffolds[murcko_scaffold] += 1

        if batch_counter % 20 == 0:
            result_write(f, scaffolds)

        batch_counter += 1
        counter += len(result)

        end_time = time.time()
        elapsed = end_time - start_time
        global_elapsed = end_time - global_start_time
        time_per_mol = global_elapsed / counter
        counter += len(result)
        log_f.write(f"Mols processed: {counter} (global per hour: {int(1 / (time_per_mol / 3600)):,})" + "\n")
        log_f.write(f"number of scaffolds: {len(scaffolds)}\n")
        log_f.flush()

        f.flush()
        start_time = time.time()

def scaffold_main():

  if len(sys.argv) > 1:
      pattern = sys.argv[1]
      glob_pattern = f"test_data/{pattern}*.txt"
      output_filename = f"nsp13_murcko_hits_{pattern}.txt"
      log_filename = f"nsp13_murcko_log_{pattern}.txt"
  else:
      glob_pattern = "test_data/*.txt"
      output_filename = f"nsp13_murcko_hits.txt"
      log_filename = f"nsp13_murcko_log.txt"

  #filenames = list(glob.glob("/proj/tropshalab/shared/REAL_Library/plaintext/*.txt"))
  filenames = list(glob.glob(glob_pattern))
  screener = UnifiedScreener(filenames, output_filename = output_filename,
              mol_function = lambda mols: [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(x)) for x in mols], log_filename = log_filename, num_workers = 30,
              delimiter = "", skip_first_line = True, smiles_position = 0, custom_write_function = scaffold_write_function,
              batch_size = 2**13)

  screener.run()



if __name__ == "__main__":
  longleaf_main()
