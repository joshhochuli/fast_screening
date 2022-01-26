from parallel_screening import ModelHolder, UnifiedScreener, SmilesIterator, SDFIterator

from rdkit.Chem import AllChem
import numpy as np
import glob


#40B library would be "/proj/tropshalab/shared/REAL_Library/plaintext/*.txt" on longleaf
filenames_to_screen = list(glob.glob("test_data/short*.txt"))

iterators = [SmilesIterator(filename, delimiter = "", skip_first_line = True, smiles_position = 0, id_position = 1) for filename in filenames_to_screen]
#if using SDF files
#iterators = [SDFIterator(filename, id_field = "ENTER HERE")]


#=======================================IF USING PICKLED QSAR MODEL=======================================================
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

screener = UnifiedScreener(iterators, #already set above
                           output_filename = "example_output_filename.txt", 
                           mol_function = model_holder.get_scores, #this shouldn't need to be changed
                           num_workers = None, #None will auto-detect cpus and use them all, specify a number otherwise
                           batch_size = 1024,  #fiddling can affect speed, don't know what's optimal
                           result_checker = lambda x: x>0.7, #function to take in result (e.g. model score) and return True/False to keep/throw away
                           )

screener.run()
#==========================================================================================================================

#=======================================IF USING PYTHON FUNCTION================================================================
'''

#must take a list of RDKit mols and return a list of scores
#example below just counts nitrogens for each molecule
def bulk_mol_function(mols):

    #operates on single mol
    def get_num_nitrogens(mol):
        nitrogen_count = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N":
                nitrogen_count += 1

        return nitrogen_count

    #operates on list of mols
    return [get_num_nitrogens(x) for x in mols]


screener = UnifiedScreener(iterators, #already set above
                           output_filename = "example_output_filename.txt", 
                           mol_function = bulk_mol_function,
                           num_workers = None, #None will auto-detect cpus and use them all, specify a number otherwise
                           batch_size = 1024,  #fiddling can affect speed, don't know what's optimal
                           result_checker = lambda x: x>0.7, #function to take in result (e.g. model score) and return True/False to keep/throw away
                           )

screener.run()
'''
#==========================================================================================================================
