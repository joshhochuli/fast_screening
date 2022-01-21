from multiprocessing import Process, Queue, cpu_count, current_process
import time
import gzip
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
import pandas as pd

class ModelHolder(object):

    def __init__(self, model_filename, descriptor_function, use_pandas = False):
        '''
        'use_pandas' will preserve names to squash the warning but hurts performance
        '''

        self.use_pandas = use_pandas

        self.model_filename = model_filename
        self.descriptor_function = descriptor_function

        with gzip.open(self.model_filename, 'rb') as f:
            self.model = pickle.load(f)

    def get_scores(self, mols):

        if self.use_pandas:
            descs = pd.DataFrame([self.descriptor_function(mol) for mol in mols])
        else:
            descs = np.array([self.descriptor_function(mol) for mol in mols])

        scores = self.model.predict_proba(descs)
        scores = scores[:,0]

        return scores

#performs all tasks (reading, computing) except writing in a single process
class UnifiedScreener(object):

    #work_function: takes in a list of RDKit mols, outputs a list of corresponding scores
    #result_checker: takes in score, outputs True if it should be kept, False if it can be thrown away. Default lambda keeps everything
    #num_workers: number of processes allowed to be spawned. If "None", detect number of cpus and use them all
    def __init__(self, filenames, output_filename, mol_function, num_workers = None, log_filename = None, result_checker = lambda x: True, delimiter = "", skip_first_line = True, smiles_position = 0, batch_size = 8192, custom_work_function = None, custom_write_function = None):

        #attempt to slaughter child processes when main process terminates
        import atexit
        atexit.register(self.__del__)

        self.delimiter = delimiter
        self.smiles_position = smiles_position
        self.skip_first_line = skip_first_line

        if num_workers == None:
            print(f"Detected {cpu_count()} cpus. Using them all.")
            num_workers = cpu_count()
        else:
            print(f"Using {self.num_workers} workers as specified")

        #build shared queue with all filenames
        self.filename_queue = Queue()
        for filename in filenames:
            extension = filename.split(".")[-1].lower()
            if extension == "smiles" or extension == "csv" or extension == "txt":
                self.filename_queue.put((filename, SmilesIterator))

        #add flag at end of queue to signal workers to exit gracefully
        self.filename_queue.put("EMPTY")

        self.result_queue = Queue()
        self.num_workers = num_workers

        self.worker_processes = []

        if custom_work_function:
            work_function = custom_work_function
        else:
            work_function = self.default_work_function

        #leave one cpu for the writing process and one for the main process
        for i in range(self.num_workers - 2):
            process = Process(target=work_function, args=(self.filename_queue, mol_function, self.result_queue, batch_size))
            self.worker_processes.append(process)

        if custom_write_function:
            write_function = custom_write_function
        else:
            write_function = self.default_write_function

        process = Process(target=write_function, args=(output_filename, self.result_queue, result_checker, log_filename))
        self.writer_process = process

    def default_write_function(self, filename, result_queue, result_checker, log_filename):

        f = open(filename, 'w')

        if log_filename == None:
            log_filename = "log.txt"
        log_f = open(log_filename, 'w')
    
        global_start_time = time.time()
        start_time = time.time()
        counter = 0
        total_counter = 1
        while True:
            result = result_queue.get(block = True, timeout = 1000)
            if result == "EMPTY":
                f.close()
                log_f.close()
                return

            

            for x in result:

                filename = x[0]
                smiles = x[1]
                mol = x[2]

                if mol == None:
                  continue

                if not result_checker(mol):
                  continue

                s = ",".join([str(i) for i in x] )
                f.write(f"{s}\n")

            counter += len(result)

            end_time = time.time()
            elapsed = end_time - start_time
            global_elapsed = end_time - global_start_time
            time_per_mol = global_elapsed / total_counter
            total_counter += len(result)
            log_f.write(f"Mols processed: {total_counter} (global per hour: {int(1 / (time_per_mol / 3600)):,})" + "\n")
            log_f.flush()

            f.flush()
            start_time = time.time()
    def __del__(self):

        for process in self.worker_processes:
            process.terminate()
            process.join()

        self.writer_process.terminate()
        self.writer_process.join()

    def dummy_mol_function(self, mol):

        fp = get_morgan_descriptor(mol)
        return np.sum(fp)

    def run(self):

        for process in self.worker_processes:
            process.start()

        time.sleep(3)
        self.writer_process.start()

        for process in self.worker_processes:
            process.join()

        self.result_queue.put("EMPTY")

    def default_work_function(self, filename_queue, mol_function, result_queue, batch_size):

        while True:
            try:
                queue_item = filename_queue.get()
            except:
                return
            if queue_item == "EMPTY":
                filename_queue.put("EMPTY")
                return

            filename, iterator_class = queue_item
            iterator = iterator_class(filename, smiles_position = self.smiles_position, delimiter = self.delimiter)
            print(f"Worker {id(current_process())} using file: {filename}")

            count = 0
            mols = []
            for mol in iterator:
                count += 1
                if count >= batch_size:

                    filenames, smiles, mols = zip(*mols)
                    scores = mol_function(mols)
                    results = list(zip(filenames, smiles, scores))
                    result_queue.put(results)
                    queue_size = result_queue.qsize()
                    count = 0
                    mols = []

                try:
                  smiles = Chem.MolToSmiles(mol)
                except:
                  smiles = None
                mols.append((filename, smiles, mol))

            #run final batch
            filenames, smiles, mols = zip(*mols)
            scores = mol_function(mols)
            results = list(zip(filenames, smiles, scores))
            result_queue.put(results)
            count = 0
            mols = []

class SplitScreener(object):

    
    #work_function: takes in an RDKit mol, outputs a score
    #result_checker: takes in score, outputs True if it should be kept, False if it can be thrown away. Default lambda keeps everything
    
    def __init__(self, filenames, output_filename, work_function, result_checker = lambda x: True, file_read_workers = 1, model_workers = 2, delimiter = "", skip_first_line = True, smiles_position = 0, batch_size = 8192):

        #attempt to slaughter children when main process terminates
        import atexit
        atexit.register(self.__del__)

        #build shared queue of all filenames
        self.filename_queue = Queue()
        for filename in filenames:
            extension = filename.split(".")[-1].lower()
            if extension == "smiles" or extension == "csv" or extension == "txt":
                self.filename_queue.put((filename, SmilesIterator))

        #put flag at end of queue to signal workers to gracefully exit
        self.filename_queue.put("EMPTY")

        self.mol_queue = Queue()
        self.result_queue = Queue()
        self.file_read_workers = file_read_workers
        self.model_workers = model_workers
        self.batch_size = batch_size

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

        while True:
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
            #print(f"READER STARTING FILE: {filename}")
            #print(filename, iterator_class)

            

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

        mol = None
        while mol == None:
            line = self.f.readline()
            if line == "":
                raise StopIteration
            if self.delimiter == "":
                s = line.split()
            else:
                s = line.split(self.delimiter)
            smiles = s[self.smiles_position]
            mol = Chem.MolFromSmiles(smiles)
            if mol == None:
                print(f"skipping line: {line}")

        return mol

    def __del__(self):
        self.f.close()
