'''
import numpy as np
import sys
sys.path.append('/home/josh/git/chemical_curation/')
#from curate import *
from joblib import Parallel, delayed

import traceback
'''

from multiprocessing import Process, Queue, cpu_count
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np

'''
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
'''

def get_morgan_descriptor(mol, radius = 2, convert_to_np = True):

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius)

    if convert_to_np:
        arr = np.array((0,))
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius), arr)
        arr = np.array(arr, dtype=np.float32)
        return arr

    return fp

#performs all tasks (reading, computing) except writing in a single process
class UnifiedScreener(object):

    def __init__(self, filenames, output_filename, mol_function, workers = 2, delimiter = "", skip_first_line = True, smiles_position = 0):

        #attempt to slaughter child processes when main process terminates
        import atexit
        atexit.register(self.__del__)

        self.filename_queue = Queue()
        for filename in filenames:
            extension = filename.split(".")[-1].lower()
            if extension == "smiles" or extension == "csv" or extension == "txt":
                self.filename_queue.put((filename, SmilesIterator))

        self.filename_queue.put("EMPTY")

        self.mol_queue = Queue()
        self.result_queue = Queue()
        self.workers = workers

        self.processes = []

        print(f"Using {self.workers} workers to do everything")
        for i in range(self.workers):
            process = Process(target=self.worker_function, args=(self.filename_queue, mol_function, self.result_queue))
            self.processes.append(process)

        process = Process(target=self.write_function, args=(output_filename, self.result_queue, self.nsp13_result_checker))
        self.processes.append(process)


    def nsp13_result_checker(self, result):

        if float(result[2]) > 0.8:
            return True
        return False

    def write_function(self, filename, result_queue, result_checker):

        f = open(filename, 'w')

        log_filename = "log.txt"
        log_f = open(log_filename, 'w')
    
        global_start_time = time.time()
        start_time = time.time()
        counter = 0
        total_counter = 1
        while True:
            result = result_queue.get(block = True, timeout = 1000)
            print(f"RESULT LENGTH: {len(result)}")

            for x in result:
                if not result_checker(x):
                    continue

                s = ",".join([str(i) for i in x] )
                f.write(f"{s}\n")
            counter += len(result)

            end_time = time.time()
            elapsed = end_time - start_time
            global_elapsed = end_time - global_start_time
            time_per_mol = global_elapsed / total_counter
            total_counter += len(result)
            log_f.write("Mol queue size: " + str(self.mol_queue.qsize()) + "\n")
            log_f.write("Result queue size: " + str(self.result_queue.qsize()) + "\n")
            log_f.write(f"Mols processed: {total_counter} (global per hour: {int(1 / (time_per_mol / 3600)):,})" + "\n")
            log_f.flush()

            f.flush()
            start_time = time.time()


        f.close()

    def __del__(self):
        print("Killing all processes!!!")

        for process in self.processes:
            process.terminate()
            process.join()

    def dummy_mol_function(self, mol):

        fp = get_morgan_descriptor(mol)
        return np.sum(fp)

    def start(self):

        print(len(self.processes))
        for process in self.processes:
            process.start()
            time.sleep(0.1)

    def worker_function(self, filename_queue, mol_function, result_queue, batch_size = 32768):

        while True:
            try:
                queue_item = filename_queue.get()
            except:
                return
            if queue_item == "EMPTY":
                return

            filename, iterator_class = queue_item
            iterator = iterator_class(filename)
            print(f"STARTING FILE: {filename}")
            print(filename, iterator_class)

            count = 0
            mols = []
            for mol in iterator:
                count += 1
                if count >= batch_size:

                    results = [(point[0], point[1], mol_function(point[2])) for point in mols]
                    result_queue.put(results)
                    count = 0
                    mols = []

                mols.append((filename, Chem.MolToSmiles(mol), mol))

            #run final batch
            results = [(point[0], point[1], mol_function(point[2])) for point in mols]
            result_queue.put(results)
            count = 0
            mols = []

   
class SplitScreener(object):

    def __init__(self, filenames, output_filename, mol_function, file_read_workers = 1, model_workers = 2, delimiter = "", skip_first_line = True, smiles_position = 0):

        #attempt to slaughter child processes when main process terminates
        import atexit
        atexit.register(self.__del__)

        self.filename_queue = Queue()
        for filename in filenames:
            extension = filename.split(".")[-1].lower()
            if extension == "smiles" or extension == "csv" or extension == "txt":
                self.filename_queue.put((filename, SmilesIterator))

        self.filename_queue.put("EMPTY")

        self.mol_queue = Queue()
        self.result_queue = Queue()
        self.file_read_workers = file_read_workers
        self.model_workers = model_workers

        self.processes = []
        print(f"Using {self.file_read_workers} workers to read files and generate mols")
        for i in range(self.file_read_workers):
            process = Process(target=self.reader_function, args=(self.filename_queue, self.mol_queue, 10))
            self.processes.append(process)

        print(f"Using {self.model_workers} workers to run models")
        for i in range(self.model_workers):
            process = Process(target=self.model_function, args=(self.mol_queue, mol_function, self.result_queue))
            self.processes.append(process)

        process = Process(target=self.write_function, args=(output_filename, self.result_queue, self.nsp13_result_checker))
        self.processes.append(process)


    def nsp13_result_checker(self, result):

        if float(result[2]) > 0.8:
            return True
        return False


    def write_function(self, filename, result_queue, result_checker):

        f = open(filename, 'w')

        log_filename = "log.txt"
        log_f = open(log_filename, 'w')
    
        global_start_time = time.time()
        start_time = time.time()
        counter = 0
        total_counter = 1
        while True:
            result = result_queue.get(block = True, timeout = 1000)
            print(f"RESULT LENGTH: {len(result)}")

            for x in result:
                if not result_checker(x):
                    continue

                s = ",".join([str(i) for i in x] )
                f.write(f"{s}\n")
            counter += len(result)

            end_time = time.time()
            elapsed = end_time - start_time
            global_elapsed = end_time - global_start_time
            time_per_mol = global_elapsed / total_counter
            total_counter += len(result)
            log_f.write("Mol queue size: " + str(self.mol_queue.qsize()) + "\n")
            log_f.write("Result queue size: " + str(self.result_queue.qsize()) + "\n")
            log_f.write(f"Mols processed: {total_counter} (global per hour: {int(1 / (time_per_mol / 3600)):,})" + "\n")
            log_f.flush()

            f.flush()
            start_time = time.time()


        f.close()

    def __del__(self):
        print("Killing all processes!!!")

        for process in self.processes:
            process.terminate()
            process.join()

    def dummy_mol_function(self, mol):

        fp = get_morgan_descriptor(mol)
        return np.sum(fp)

    def start(self):

        print(len(self.processes))
        for process in self.processes:
            process.start()
            time.sleep(0.1)

    def model_function(self, mol_queue, mol_function, result_queue):

        while True:
            while True:
                try:
                    data = mol_queue.get(block = True, timeout = 0.1)
                except:
                    continue
                if data == "EMPTY":
                    print("MODEL WORKER GOT EMPTY, RETURNING")
                    return
                else:
                    break
                #data = mol_queue.get(block = True, timeout = 100)
            print("SUCCESSFUL READ")
            results = [(point[0], point[1], mol_function(point[2])) for point in data]
            result_queue.put(results)

    def reader_function(self, filename_queue, mol_queue, max_queue_size = 10, batch_size = 32768):

        print("ENTERING READER")
        while True:
            print("HERE")
            try:
                queue_item = filename_queue.get()
            except:
                return
            print(queue_item)
            try:
                filename, iterator_class = queue_item
            except:
                print("!!!!!")
                print(queue_item)
                print("!!!!!")
            iterator = iterator_class(filename)
            print(f"READER STARTING FILE: {filename}")
            print(filename, iterator_class)

            

            count = 0
            mols = []
            for mol in iterator:
                count += 1
                if count >= batch_size:
                    queue_size = mol_queue.qsize()
                    while queue_size > max_queue_size:
                        time.sleep(5)
                        queue_size = mol_queue.qsize()
                    print("PUTTING")
                    mol_queue.put(mols)
                    mols = []
                    count = 0

                mols.append((filename, Chem.MolToSmiles(mol), mol))
    
class SDFIterator():
    pass

class SmilesIterator():

    def __init__(self, filename, skip_first_line = True, delimiter = ",", smiles_position = 0):

        self.filename = filename
        self.delimiter = delimiter
        self.smiles_position = smiles_position
        f = open(filename, 'r')

        if skip_first_line:
            f.readline()

        self.f = f

    def __iter__(self):
        return self

    def __next__(self):

        line = self.f.readline()
        if line == "":
            raise StopIteration
        if self.delimiter == "":
            s = line.split()
        else:
            s = line.split(self.delimiter)
        smiles = s[self.smiles_position]
        mol = Chem.MolFromSmiles(smiles)
        return mol

    def __del__(self):
        self.f.close()

'''
class Screener(object):

    def __init__(self, input_filename, output_filename, molecule_description_function = lambda x: x, batch_size =1024, num_workers = 1):

'''

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
