import numpy as np
from rdkit import Chem
import sys
sys.path.append('/home/josh/git/chemical_curation/')
from curate import *
from joblib import Parallel, delayed

import traceback

#trying to run the rdkit fingerprint functions in parallel complains about pickling boost functions
#this stackoverflow magic sidesteps it
#although it might cause a lot of overhead with the import every time? or maybe it's cached?
class Wrapper(object):
    def __init__(self, method_name, module_name):
        self.method_name = method_name
        self.module_name = module_name

    def __call__(self, *args, **kwargs):
        module = __import__(self.module_name, globals(), locals(), [self.method_name,])
        method = getattr(module, self.method_name)
        return method(*args, **kwargs)

def parallel_screen_smiles(input_filename, id_location, model, fingerprint_function, output_filename, batch_size = 1024, num_workers = 4):

    f = open(input_filename, 'r')
    o = open(output_filename, 'a+')

    smiles_list = []
    count = 0
    for line in f:
        smiles_list.append(line)
        count = count + 1
        if len(smiles_list) == batch_size:

            active_ids, active_smiles = predict_batch(smiles_list, model,
                id_info = id_location, fingerprint_function = fingerprint_function, 
                reading_function = smiles_to_fp, num_workers = num_workers)

            for i in range(len(active_ids)):
                o.write(f"{active_ids[i]},{active_smiles[i]}\n")
            smiles_list  = []
            print(f"{count} molecules processed")

    if len(smiles_list) > 0:

        active_ids, active_smiles = predict_batch(smiles_list, model,
                id_info = id_location, fingerprint_function = fingerprint_function, 
                reading_function = smiles_to_fp, num_workers = num_workers)

        for i in range(len(active_ids)):
            o.write(f"{active_ids[i]},{active_smiles[i]}\n")
        print(f"{count} molecules processed")



def parallel_screen_sdf(input_filename, id_name, model, fingerprint_function, output_filename, batch_size = 1024, num_workers = 4):

    f = open(input_filename, 'r')
    o = open(output_filename, 'a+')

    mols = []
    curr_mol = []
    count = 0
    for line in f:
        if "$$$$" in line:
            count = count + 1
            mol_string = "".join(curr_mol)
            curr_mol = []
            mols.append(mol_string)
            if len(mols) == batch_size:
                active_ids, active_smiles = predict_batch(mols, model,
                    id_name = id_name, fingerprint_function = fingerprint_function, reading_function = mol_block_to_fp, num_workers = num_workers)
                for i in range(len(active_ids)):
                    o.write(f"{active_ids[i]},{active_smiles[i]}\n")
                mols  = []
                print(f"{count} molecules processed")

        else:
            curr_mol.append(line)

    if len(mols) > 0:
        active_ids, active_smiles = predict_batch(mols, model,
            id_name = id_name, fingerprint_function = fingerprint_function, reading_function = mol_block_to_fp, num_workers = num_workers)
        for i in range(len(active_ids)):
            o.write(f"{active_ids[i]},{active_smiles[i]}\n")
        mols  = []
        print(f"{count} molecules processed")

#rdkit's MolFromMolBlock doesn't seem to read in properties
def get_property_from_mol_block(mol_block, property_name):

    next_line = False
    lines = mol_block.split('\n')
    for line in lines:
        if next_line:
            return line.strip()
        if property_name in line:
            next_line = True

    return None

def mol_block_to_fp(mol_block, fingerprint_function, id_name):

    try:
        id_val = get_property_from_mol_block(mol_block, id_name)
        mol = Wrapper("MolFromMolBlock", "rdkit.Chem")(mol_block)
        curated_mol = Mol.from_rdkit_mol(mol, precise_activities = 1)
        mol = curated_mol.mol
        smiles = Chem.MolToSmiles(mol)
        if mol == None:
            return None
        fp = fingerprint_function(mol)
    except Exception as e:
        print(e)
        return None
    return fp, id_val, smiles

#take in whole line of smiles file
#id_location == 0 will use first token as id and second as smiles
#id_location == 1 will use first token as smiles and second as id
def smiles_to_fp(smiles_line, fingerprint_function, id_location):

    s = smiles_line.split(",")
    if id_location == 0:
        id_val = s[0].strip()
        smiles = s[1].strip()
    elif id_location == 1:
        id_val = s[1].strip()
        smiles = s[0].strip()
    else:
        raise Exception(f"id_location must be 0 or 1 ({id_location} provided)")

    try:
        mol = Wrapper("MolFromSmiles", "rdkit.Chem")(smiles)
        curated_mol = Mol.from_rdkit_mol(mol, precise_activities = 0)
        smiles = Chem.MolToSmiles(curated_mol.mol)
        if curated_mol.mol == None:
            return None
        fp = fingerprint_function(curated_mol.mol)
    except Exception as e:
        print(e)
        return None

    return fp, id_val, smiles


def predict_batch(batch, model, id_info, fingerprint_function, reading_function, num_workers):

    if reading_function == smiles_to_fp:
        data = Parallel(n_jobs = num_workers)(delayed(reading_function)(x, fingerprint_function, id_info) for x in batch)
    elif reading_function == mol_block_to_fp:
        data = Parallel(n_jobs = num_workers)(delayed(reading_function)(x, fingerprint_function, id_info) for x in batch)
    data = [x for x in data if x != None]
    fps = np.array([x[0] for x in data])
    ids = np.array([x[1] for x in data])
    smiles = np.array([x[2] for x in data])
    activities = model.predict(fps)

    i = np.where(activities == 1)[0]

    if len(i) > 0:
        active_id_vals = ids[i]
        active_smiles_list = smiles[i]

    else:
        return [], []

    return active_id_vals, active_smiles_list

def main():

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

if __name__ == "__main__":
    main()
