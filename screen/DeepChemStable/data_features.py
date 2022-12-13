from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import autograd.numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit import Chem
import rdkit.Chem.rdPartialCharges as rdPartialCharges
import tensorflow as tf

degrees = [1, 2, 3, 4, 5]

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, add_Gasteiger):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B',   'H',  'Unknown']) +  # H?
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()] + 
                   [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +  
                    one_of_k_encoding_unk(atom.GetHybridization(), [           
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2]) + [add_Gasteiger])

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a, 0.0))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

import numpy as np
class MolGraph(object):
    def __init__(self):
        self.nodes = {} # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))

    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i : [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            new_nodes.extend(cur_nodes)

        self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n : i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']
    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]

def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_smiles(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph

def graph_from_smiles(smiles):
    graph = MolGraph()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    atoms_by_rd_idx = {}
    
    rdPartialCharges.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
        add_Gasteiger = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(add_Gasteiger) or np.isinf(add_Gasteiger):
            add_Gasteiger = 0.0
        new_atom_node = graph.new_node('atom', features=atom_features(atom, add_Gasteiger), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph


@memoize
def array_rep_from_smiles(smiles):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    molgraph = graph_from_smiles_tuple(smiles)
    arrayrep = {'atom_features' : molgraph.feature_array('atom'),
                'bond_features' : molgraph.feature_array('bond'),
                'atom_list'     : molgraph.neighbor_list('molecule', 'atom'), # List of lists.
                'rdkit_ix'      : molgraph.rdkit_ix_array()}  # For plotting only.
    for degree in degrees:
        arrayrep[('atom_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'atom'), dtype=int)
        arrayrep[('bond_neighbors', degree)] = \
            np.array(molgraph.neighbor_list(('atom', degree), 'bond'), dtype=int)
    return arrayrep


# read data
import os
import csv
import numpy as np
import itertools as it

def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return list(map(np.array, data))

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]


def trans(substances):
    substance_atoms = []
    for substance_i, atoms_i in enumerate(substances['atom_list']):
         substance_atoms += [ [substance_i, atoms_ij] for atoms_ij in atoms_i]
    substance_atoms = np.array(substance_atoms)
    n_atoms = substance_atoms.shape[0]
    
    substance_atom_indices = substance_atoms
    substance_atom_values = tf.fill(tf.expand_dims(tf.cast(n_atoms, tf.int32), 0), 1.0)
    substance_atom_shape = [substance_i+1, n_atoms]
    substance_atoms_tensor = tf.SparseTensor(substance_atom_indices, 
                                             substance_atom_values, substance_atom_shape)
    
    substances['substance_atoms'] = substance_atoms_tensor
    
    # features
    substances["atom_features"] = substances["atom_features"].astype(np.float32)
    substances["bond_features"] = substances["bond_features"].astype(np.float32)
    
    # rdkit_ix  and compound id
    compounds_rdkit_ix = []
    for substance_i, atoms_i in enumerate(substances["atom_list"]):
        atom_rdkit_ix = substances["rdkit_ix"][atoms_i]
        compounds_rdkit_ix += [[substance_i, atom_rdkit_ix_i] for atom_rdkit_ix_i in atom_rdkit_ix]
    compounds_rdkit_ix = np.array(compounds_rdkit_ix)
    substances["compounds_rdkit_ix"] = compounds_rdkit_ix
            
    # neighbors
    for degree in degrees:
        
        atom_neighbors = substances[('atom_neighbors', degree)]
        if atom_neighbors == []:
            print(temp)
        substances['atom_neighbors_{}'.format(degree)] = atom_neighbors
        substances.pop(('atom_neighbors', degree)) 
    
    for degree in degrees:
        bond_neighbors = substances[('bond_neighbors', degree)]
        substances['bond_neighbors_{}'.format(degree)] = bond_neighbors
        substances.pop(('bond_neighbors', degree))
        
    # rnn raw input with atom list
    N_compounds = max(substances["compounds_rdkit_ix"][:, 0])+1
    N_max_seqlen = max(substances["compounds_rdkit_ix"][:, 1]) + 1
    rnn_raw_input = np.zeros((N_compounds, N_max_seqlen), dtype=np.int64) + n_atoms
    #rnn_raw_input = np.zeros((N_compounds, N_max_seqlen), dtype=np.int64) 
    for i, atoms_i in enumerate(substances['atom_list']):
        for j, a_j in enumerate(atoms_i):
            rnn_raw_input[i, j] = a_j
    substances["rnn_raw_input"] = rnn_raw_input
    
    return substances
