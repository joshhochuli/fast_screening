import unittest
import numpy as np
import glob
from rdkit.Chem import AllChem

from parallel_screening import ModelHolder, UnifiedScreener, SDFIterator, SmilesIterator

from scoring_function import get_nsp13_structural_reward




class TestAll(unittest.TestCase):

    def test_SmilesIterator(self):

        filename = "test_data/10.txt"

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

        filename = "test_data/test.sdf"

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

    '''
    def test_Screener(self):

        filenames = list(glob.glob("test_data/a*.txt"))
        screener = UnifiedScreener(filenames, output_filename = "output.txt", mol_function = get_nsp13_structural_reward, num_workers = 10, delimiter = "", skip_first_line = True, smiles_position = 0)

        screener.run()
    '''


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




    '''
    def test_normal(self):


        model = get_model()

        #molecule_by_molecule_screen('/home/josh/git/sars-cov-mpro/datasets/curated_data/drugbank.sdf', model = model, output_filename = 'test.csv', num_workers = 2)
        parallel_screen_sdf('/home/josh/git/sars-cov-mpro/datasets/curated_data/drugbank.sdf', model = model, output_filename = 'test.csv', num_workers = 12, batch_size = 4096)

        exit()

        filenames = ["data/Enamine_advanced_collection_202002.sdf", 
                    "data/Enamine_Hit_Locator_Library_300115cmpds_20200110.sdf",
                    "data/Enamine_functional_collection_43057cmpds_20200220.sdf",
                    "data/Enamine_premium_collection_202002.sdf"]

        output_names = ["output/advanced_hits.csv",
                        "output/hll_hits.csv",
                        "output/functional_hits.csv",
                        "output/premium_hits.csv"]

        for i in range(len(filenames)):
            filename = filenames[i]
            output_filename = output_names[i]
            print(filename)

            molecule_by_molecule_screen(filename, model = model, output_filename = output_filename)


    '''

if __name__ == "__main__":
    unittest.main()
