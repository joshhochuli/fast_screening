import unittest
import numpy as np
import glob
from rdkit.Chem import AllChem

from parallel_screening import ModelHolder, UnifiedScreener, SDFIterator, SmilesIterator

from scoring_function import get_nsp13_structural_reward

class TestAll(unittest.TestCase):

    def test_SmilesIterator(self):

        filename = "test_data/short_xaa.txt"

        iterator = SmilesIterator(filename = filename, skip_first_line = True, delimiter = "", smiles_position = 0)

        print("Testing no supplied id_position")
        for i, data in enumerate(iterator):
            if i > 3:
                break
            print(data)

        print("Testing supplied id_position")
        iterator = SmilesIterator(filename = filename, skip_first_line = True, delimiter = "", smiles_position = 0, id_position = 1)

        for i, data in enumerate(iterator):
            if i > 3:
                break
            print(data)


    def test_SDFIterator(self):

        filename = "test_data/short_xaa.sdf"

        print("Testing no supplied id_field")
        iterator = SDFIterator(filename)

        for i, data in enumerate(iterator):
            if i > 3:
                break
            print(data)
 
        print("Testing supplied id_field")
        iterator = SDFIterator(filename, id_field = "ID")

        for i, data in enumerate(iterator):
            if i > 3:
                break
            print(data)

 
    def test_qsar_Screener_smiles(self):

        filenames = list(glob.glob("test_data/short*.txt"))

        def calcfp(mol,funcFPInfo=dict(radius=3, nBits=2048, useFeatures=False, useChirality=False)):

            fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo))
            return fp

        model_holder = ModelHolder(model_filename = "test_data/qsar_model.pgz", descriptor_function = calcfp)
        iterators = [SmilesIterator(filename, delimiter = "", skip_first_line = True, smiles_position = 0) for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "qsar_screener_test.txt", mol_function = model_holder.get_scores, num_workers = None, batch_size = 1024, result_checker = lambda x: x>0.7)

        screener.run()

    def test_qsar_Screener_sdf(self):

        filenames = list(glob.glob("test_data/short*.sdf"))

        def calcfp(mol,funcFPInfo=dict(radius=3, nBits=2048, useFeatures=False, useChirality=False)):

            fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo))
            return fp

        model_holder = ModelHolder(model_filename = "test_data/qsar_model.pgz", descriptor_function = calcfp)
        iterators = [SDFIterator(filename, id_field = "ID") for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "qsar_screener_test.txt", mol_function = model_holder.get_scores, num_workers = None, batch_size = 1024, result_checker = lambda x: x>0.7)

        screener.run()

    def test_pharmacophore_Screener_smiles(self):

        filenames = list(glob.glob("test_data/short*.txt"))

        def bulk_mol_function(mols):

            return [get_nsp13_structural_reward(x) for x in mols]

        iterators = [SmilesIterator(filename, delimiter = "", skip_first_line = True, smiles_position = 0, id_position = 1) for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "pharmacophore_smiles_test_output.txt", mol_function = bulk_mol_function)

        screener.run()

    def test_pharmacophore_Screener_sdf(self):

        filenames = list(glob.glob("test_data/short*.sdf"))

        def bulk_mol_function(mols):

            return [get_nsp13_structural_reward(x) for x in mols]

        iterators = [SDFIterator(filename, id_field = "ID") for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "pharmacophore_sdf_test_output.txt", mol_function = bulk_mol_function)

        screener.run()


    def test_arbitrary_Screener_smiles(self):

        filenames = list(glob.glob("test_data/short*.txt"))

        def bulk_mol_function(mols):

            def get_num_nitrogens(mol):
                nitrogen_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "N":
                        nitrogen_count += 1

                return nitrogen_count

            return [get_num_nitrogens(x) for x in mols]

        iterators = [SmilesIterator(filename, delimiter = "", skip_first_line = True, smiles_position = 0, id_position = 1) for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "arbitrary_smiles_test_output.txt", mol_function = bulk_mol_function)

        screener.run()

    def test_arbitrary_Screener_sdf(self):

        filenames = list(glob.glob("test_data/short*.sdf"))

        def bulk_mol_function(mols):

            def get_num_nitrogens(mol):
                nitrogen_count = 0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == "N":
                        nitrogen_count += 1

                return nitrogen_count

            return [get_num_nitrogens(x) for x in mols]

        iterators = [SDFIterator(filename, id_field = "ID") for filename in filenames]
        screener = UnifiedScreener(iterators, output_filename = "arbitrary_sdf_test_output.txt", mol_function = bulk_mol_function)

        screener.run()

if __name__ == "__main__":
    unittest.main()
