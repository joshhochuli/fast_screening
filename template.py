from parallel_screening import ModelHolder, UnifiedScreener

from rdkit.Chem import AllChem
import numpy as np
import glob


#40B library would be "/proj/tropshalab/shared/REAL_Library/plaintext/*.txt" on longleaf
filenames_to_screen = list(glob.glob("test_data/*.txt"))

#location of pickled model, must be .pgz
#if you run and get warnings about Classifier versions, its best to update the environment to match exactly
#'conda install scikit-learn={version_from_warning}'
#e.g. 'conda install scikit-learn=0.24.2'
model_filename = "test_data/qsar_model.pgz"

#need a function to take in an rdkit molecule and return the descriptor exactly as its used in the model
def get_descriptor(mol,funcFPInfo=dict(radius=3, nBits=2048, useFeatures=False, useChirality=False)):

    fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo))
    return fp


#change both model_filename
model_holder = ModelHolder(model_filename = model_filename, 
                           descriptor_function = get_descriptor)


screener = UnifiedScreener(filenames_to_screen, #variable set above
                           output_filename = "example_output_filename.txt", 
                           mol_function = model_holder.get_scores, #this shouldn't need to be changed
                           num_workers = None, #None will auto-detect cpus and use them all, specify a number otherwise
                           delimiter = "", #default for 40B library
                           skip_first_line = True, #default for 40B library
                           smiles_position = 0, #default for 40B library
                           batch_size = 1024,  #fiddling can affect speed, don't know what's optimal
                           result_checker = lambda x: x>0.7, #function to take in result (e.g. model score) and return True/False to keep/throw away
                           )

screener.run()



