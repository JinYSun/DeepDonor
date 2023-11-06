# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 17:15:26 2021

@author: BM109X32G-10GPU-02
"""
# Native imports
from subprocess import Popen
from subprocess import PIPE
from os import remove
import pandas as pd 
# rdkit imports
from rdkit import Chem
from rdkit.Chem import Descriptors

def rdkit_descriptors(mol):
	"""
	Get the molecular descriptors by iterating over all possible descriptors.

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	
	# Iterate over all descriptors, get their functions (func) and apply to the molecule object
	for name,func in Descriptors.descList:
		ret_dict[name] = func(mol)
	return(ret_dict)

def cdk_descriptors(mol,temp_f_smiles_name="tempsmiles.smi",temp_f_cdk_name="tempcdk.txt"):
	"""
	Get the molecular descriptors in cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
	temp_f_smiles_name : str
		Where to temporarily store the smiles
	temp_f_cdk_name : str
		Where to temporarily store the cdk file
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}

	# Get the SMILES from the rdkit object
	smiles = Chem.MolToSmiles(mol,1)
		
	# Write SMILES to a temp file
	temp_f_smiles = open(temp_f_smiles_name,"w")
	temp_f_smiles.write("%s temp" % smiles)
	temp_f_smiles.close()
	
	# Get the features, descriptors indicate the main class of descriptors
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="topological"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="geometric"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="constitutional"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="electronic"))
	ret_dict.update(call_cdk(infile=temp_f_smiles_name,outfile=temp_f_cdk_name,descriptors="hybrid"))
	
	# Remove the temporary files
	remove(temp_f_smiles_name)
	remove(temp_f_cdk_name)

	return(ret_dict)


def call_cdk(infile="",outfile="",descriptors=""):
	"""
	Make a system call to cdk to get descriptors

    Parameters
    ----------
    infile : str
        String indicating the filename for the input SMILES
	outfile : str
		String indicating the filename for the output values
	descriptors : str
        Main class of molecular descriptors to retrieve
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cmd = "java -jar CDKDescUI-1.4.6.jar -b %s -a -t %s -o %s" % (infile,descriptors,outfile)
	p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
	out = p.communicate()
	return(parse_cdk_file(outfile))

def parse_cdk_file(file):
	"""
	Parse the cdk output files

    Parameters
    ----------
    file : str
        String indicating the filename for the output of cdk
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	cdk_file = open(file).readlines()
	cols = cdk_file[0].strip().split()[1:]
	feats = cdk_file[1].strip().split()[1:]
	return(dict(zip(cols, feats)))

def getf(mol,progs=["rdkit"]):
	"""
	Main function that makes the call to rdkit or cdk

    Parameters
    ----------
    mol : object
        An rdkit molecules object.
	progs : list
		List containing the tools to use for calculating molecular descriptors. Use "rdkit" and or "cdk"
		
    Returns
    -------
    dict
		Dictionary that contains the molecular descriptor names as keys and their respective values 
    """
	ret_dict = {}
	if "rdkit" in progs: ret_dict["rdkit"] = rdkit_descriptors(mol)
	if "cdk" in progs: ret_dict["cdk"] = cdk_descriptors(mol)
	return(ret_dict)
def screen (filename):
    file = pd.read_csv(filename,header=None)
    fi = file.iloc[1:,1]
    f = []
    for ind,smi in enumerate( fi):
         
         mol = Chem.MolFromSmiles(smi)
         a = getf(mol)
         b=a['rdkit'] 
         if b['MolLogP']<40.7 and b['MolLogP']>8.76: #and b['MolLogP']>8.76
             if b['MolWt']<1967 and b['MolWt']>485: #and b['MolWt']>485
                 if b['NOCount']<6:
                     if    b['NumHAcceptors']<15  and b['NumHAcceptors']>3:
                       if  b['NumHDonors']<2:
                         if    b['NumHeteroatoms']<17 and b['NumHeteroatoms']>5.7: #and b['NumHeteroatoms']>5.7
                           #if      b['NumRadicalElectrons']<7:
                             if        b['NumRotatableBonds']<55 and b['NumRotatableBonds']>13.3:
                    
                                 if            b['RingCount']<13 and b['RingCount']>2.85:
                                   if              b['NumAromaticHeterocycles']<11.55  :
                                     if                b['NumAromaticRings']<12.6 :
                                                      
                                                 if 'P' not in smi:
                                                             print(ind)
                                                             file.iloc[ind,:]
                                                             f.append(smi)
                                                         
    res= pd.DataFrame(f)
    #res = pd.concat([file,f1],axis=1)
    res.to_csv(r"screen1.smi")  

if __name__ == "__main__":
    screen()


