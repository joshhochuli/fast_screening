import gzip
import pickle
import pandas as pd
import numpy as np
import networkx as nx

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


class CompositeRewarder:

    def __init__(self):

        self.qsar_scorer = QSARScorer()

        #might have to enforce that these functions return within [0,1]?
        self.reward_functions = [get_nsp13_structural_reward, self.qsar_scorer.get_average_score]

        self.composition_function = np.prod

    def get_reward(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            return 0

        scores = []
        for reward_function in self.reward_functions:
            score = reward_function(mol)
            scores.append(score)

        composite_score = float(self.composition_function(scores))
        return composite_score

def get_nsp13_structural_reward(mol):

    full_pattern_smarts = "a1[aD2]a([F,Cl,Br,I])[aD2][aD2]a1[A,a]C=O"
    halide_ring_smarts = "a1[aD2]a([F,Cl,Br,I])[aD2][aD2]a1"
    ring_carbonyl_smarts = "a1[aD2]a[aD2][aD2]a1[A,a]C=O"

    full_pattern = Chem.MolFromSmarts(full_pattern_smarts)
    halide_ring = Chem.MolFromSmarts(halide_ring_smarts)
    ring_carbonyl = Chem.MolFromSmarts(ring_carbonyl_smarts)


    '''
    Chem.Draw.ShowMol(full_pattern)
    Chem.Draw.ShowMol(halide_ring)
    Chem.Draw.ShowMol(ring_carbonyl)
    '''

    rot_bond_num_threshold = 6
    density_smarts = 'a1aaaaa1'

    halide = Chem.MolFromSmarts("[F,Cl,Br,I]")
    carbonyl = Chem.MolFromSmarts("C=O")
    ring = Chem.MolFromSmarts("a1[aD2]a[aD2][aD2]a1")

    anchor_smarts = None

    #lazy reward normalization
    max_reward = 7

    if mol.HasSubstructMatch(full_pattern):
        nominal_reward = 5

        anchor_smarts = full_pattern_smarts

    elif mol.HasSubstructMatch(halide_ring) and mol.HasSubstructMatch(carbonyl):
        nominal_reward = 4

        anchor_smarts = halide_ring_smarts


    elif mol.HasSubstructMatch(ring_carbonyl) and mol.HasSubstructMatch(halide):
        nominal_reward = 4

        anchor_smarts = ring_carbonyl_smarts

    elif mol.HasSubstructMatch(halide_ring):
        nominal_reward = 3

        anchor_smarts = halide_ring_smarts

    elif mol.HasSubstructMatch(ring_carbonyl):
        nominal_reward = 3

        anchor_smarts = ring_carbonyl_smarts

    else:
        individual_scores = [mol.HasSubstructMatch(halide),
                            mol.HasSubstructMatch(carbonyl),
                            mol.HasSubstructMatch(ring)]

        if mol.HasSubstructMatch(carbonyl):
            anchor_smarts = 'C=O'

        nominal_reward = sum(individual_scores)

    rot_bond_num = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if rot_bond_num <= rot_bond_num_threshold:
        nominal_reward += 1

    if anchor_smarts:
        if check_linker(mol, anchor_smarts, (-1,), density_smarts, (1, 2, 3, 4, 5, 6), (2, 5)):
            nominal_reward += 1

    return nominal_reward / max_reward


def mol2graph(mol):
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def calc_shortest_path_len(graph, src_group, dst_group):
    shortest_path_len = None
    for src in src_group:
        for dst in dst_group:
            if nx.has_path(graph, src, dst):
                path_len = nx.shortest_path_length(graph, source=src, target=dst)
                if shortest_path_len and path_len < shortest_path_len:
                    shortest_path_len = path_len
    return shortest_path_len


def remaining_graph(graph, match_1, growth_match_1, match_2, growth_match_2):
    tmp_graph = graph.copy()
    growth_set_1, growth_set_2 = set(growth_match_1), set(growth_match_2)
    rm_nbunch = (set(match_1) - growth_set_1) | (set(match_2) - growth_set_2)
    tmp_graph.remove_nodes_from(rm_nbunch)
    
    rm_ebunch = []
    for edge in tmp_graph.edges:
        if growth_set_1.issuperset(edge) or growth_set_2.issuperset(edge):
            rm_ebunch.append(edge)
    tmp_graph.remove_edges_from(rm_ebunch)
    return tmp_graph


def check_linker(mol, anchor_smarts, anchor_growth_idxs, density_smarts, density_growth_idxs,
                 linker_len_range):
    graph = mol2graph(mol)

    anchor_patt = Chem.MolFromSmarts(anchor_smarts)
    density_patt = Chem.MolFromSmarts(density_smarts)

    # Simplified for our specific case and speed
    # (uniquified, growth region are invariant wrt internal symmetry)
    anchor_uniquified_matches = mol.GetSubstructMatches(anchor_patt)
    density_uniquified_matches = mol.GetSubstructMatches(density_patt)
    for anchor_match in anchor_uniquified_matches:
        for density_match in density_uniquified_matches:

            if not (set(anchor_match) & set(density_match)): # it can be a separate +1 reward
                anchor_growth_match = (anchor_match[i] for i in anchor_growth_idxs)
                density_growth_match = (density_match[i - 1] for i in density_growth_idxs)
                tmp_graph = remaining_graph(graph, anchor_match, anchor_growth_match,
                                            density_match, density_growth_match)

                shortest_path_len = calc_shortest_path_len(graph, anchor_match,
                                                           density_growth_match)
                if shortest_path_len and (shortest_path_len >= linker_len_range[0]) and (shortest_path_len <= linker_len_range[1]):
                    return True
    return False


#made a class so models will only be loaded one time
#returns a score from 0 to 1, with 1 being good and 0 being bad
#good and bad is controlled by target_score variable below (changed depending on exact model)
class QSARScorer:

    #model_filename, name, target_score
    nsp13_pk_models = [
            ("dataset-06-plasma-protein-binding-binary-unbalanced-morgan_RF.pgz", "Plasma Protein Binding", 0),
            ("dataset-09-microsomal-clearance-binary-unbalanced-morgan_RF.pgz", "Microsomal Clearance", 0),
            ("dataset-02A-microsomal-half-life-subcellular-30-min-binary-unbalanced-morgan_RF.pgz", "Microsomal Half-Life", 1),
            ("dataset-01D-hepatic-stability-60-min-binary-balanced-morgan_RF.pgz", "Hepatic Stability", 1),
            ("dataset-05A-CACO2-binary-unbalanced-morgan_RF.pgz", "CACO-2 Permeability", 1),
            ("dataset-03-renal-clearance-binary-unbalanced-0.5-threshold-morgan_RF.pgz", "Renal Clearance", 0),
            ("dataset-02B-microsomal-half-life-30-min-binary-unbalanced-morgan_RF.pgz", "Microsomal Half-Life", 1)
            ]

    def calcfp(self, mol,funcFPInfo=dict(radius=3, nBits=2048, useFeatures=False, useChirality=False)):

        #super lazy caching
        if mol == self.last_mol:
            return self.last_fp
        else:
            fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, **funcFPInfo))
            self.last_mol = mol
            self.last_fp = fp

        return fp

    def __init__(self):

        self.last_mol = None
        self.last_fp = None

        self.models = {}

        for model, name, target_score in self.nsp13_pk_models:
            full_filename = f"qsar_models/{model}"
            with gzip.open(full_filename, 'rb') as f:
                model = pickle.load(f)

            self.models[name] = (model, target_score)

    def get_score(self, mol, model_name):

        if model_name not in self.models:
            raise Exception(f"Provided model name '{model_name}' not loaded")

        model, target_score = self.models[model_name]

        fp = self.calcfp(mol).reshape(1,-1)

        prediction = model.predict_proba(fp)
        return float(prediction[0][target_score])

    def get_all_scores(self, mol):

        scores = {}
        for model_name in self.models.keys():
            score = self.get_score(mol, model_name)
            scores[model_name] = score

        return scores

    def get_average_score(self, mol):

        average_score = float(np.average(list(self.get_all_scores(mol).values())))
        return average_score
