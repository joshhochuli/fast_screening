import unittest
import glob

from parallel_screening import *

from scoring_function import get_nsp13_structural_reward




class TestWorking(unittest.TestCase):

    def test_SmilesIterator(self):

        filename = "test_data/smile_all_520.txt"
        iterator = SmilesIterator(filename = filename, skip_first_line = True, delimiter = "", smiles_position = 0)

        for i, mol in enumerate(iterator):
            if i > 10:
                break
            print(mol)

    def test_Screener(self):

        filenames = list(glob.glob("test_data/*.txt"))
        screener = UnifiedScreener(filenames, output_filename = "screener_test.txt", mol_function = get_nsp13_structural_reward, workers = 31, delimiter = "", skip_first_line = True, smiles_position = 0)

        screener.start()


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
