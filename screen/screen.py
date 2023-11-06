# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:42:02 2023

@author: Jinyu Sun
"""
import wget
import basic
from sascore.src import sascorer
from scscore.scscore import standalone_model_numpy
from DeepChemStable import predict
import zstandard

wget.download(r'https://zenodo.org/records/10071090/files/Gen dataset.smi.zst?download=1','Gen.smi.zst')
dctx = zstandard.ZstdDecompressor()
with open('Gen.smi.zst', 'rb') as ifh, open('Gen.smi', 'wb') as ofh:
        dctx.copy_stream(ifh, ofh)
       
# screen with basic properties
basic.screen('Gen.smi')

#screen with SAscore
sascorer.screen('screen1.smi')

#screen with SCscore

standalone_model_numpy.screen('screen2.csv')

#screen with DeepChemStable
predict.screen('screen3.csv')