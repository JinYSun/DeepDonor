# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import inspect
import pandas as pd
import numpy as np
from data_features import *
from model import *
from utils import *
import tensorflow as tf
import pickle
tf.compat.v1.disable_eager_execution()
data_file = r"J:\screendonor\DeepChemStable\genscreennew55.csv"
N_test = 17675

data_test = pd.read_csv(data_file)#.iloc[:1000,:]
substance_names = data_test["substance_id"]


# load trained variables
fingerprint_variables_file=open(r'./DeepChemStable_model/fingerprint_variables.bin','rb')
fingerprint_variables_file.seek(0)
fingerprint_variables_array =pickle.load(fingerprint_variables_file)

prediction_variables_file=open(r'./DeepChemStable_model/prediction_variables.bin','rb')
prediction_variables_file.seek(0)
prediction_variables_array =pickle.load(prediction_variables_file)

fingerprint_variables = {}
prediction_variables = {}
for key in fingerprint_variables_array.keys():
    fingerprint_variables[key] =tf.convert_to_tensor(fingerprint_variables_array[key],  dtype=tf.float32)
for key in prediction_variables_array.keys():
    prediction_variables[key] =tf.convert_to_tensor(prediction_variables_array[key],  dtype=tf.float32)


# parameters
task_params = dict(target_name = "label",
                   data_file = data_file,
                   N_test = N_test)

train_params = dict(
                    n_batches = 5,
                    log_init_scale= -2,
                    log_learning_rate=-4,
                    log_b1 = -3,
                    log_b2 = -2)

model_params = dict(
                    fp_length = 200,
                    fp_depth = 4,
                    fp_width = 30,
                    h1_size =100,
                    l2_penalty = 0.0001, 
                    pos_weight = 1.5)
model_params['prediction_layer_sizes'] = [model_params['fp_length'], model_params['h1_size'], 1]

# read data
testdata = load_data(
    task_params['data_file'], [task_params["N_test"]],
    input_name='smiles', target_name=task_params['target_name'])
test_inputs,  test_labels = testdata[0]
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)
test_labels = tf.reshape(test_labels, [-1, 1])

test_substances = array_rep_from_smiles(tuple(test_inputs))
test_substances =  trans(test_substances)


# test
test_fps, test_rnn_embedding,  test_atom_activations = build_neural_fps_network(test_substances, fingerprint_variables, model_params)
test_fps, test_att_prob = build_attention(test_rnn_embedding, test_substances, model_params, is_training=False)
test_normed_predictions = build_convolution_prediction_network(test_fps,
                                                                prediction_variables, model_params)
test_predictions, test_loss = build_loss_network(test_normed_predictions, 
                                                   test_labels, fingerprint_variables, prediction_variables, model_params)


with tf.compat.v1.Session() as sess:
            
    test_preds, test_targets, test_atom_activations_vis, test_att_prob_vis = sess.run(fetches=[test_predictions, test_labels, test_atom_activations, test_att_prob])
	
    probs = sigmoid(test_preds)
    labels = np.where(probs>=0.5, "Unstable", "Stable")
    
    out_names = []
    out_probs = []
    out_labels = []
    print("substance_id", "Probability", "Label")
    for i in range(len(probs)):
        out_names.append(str(substance_names[i]))
        out_probs.append(round(float(probs[i][0]), 3))
        out_labels.append(str(labels[i][0]))
        print(out_names[i], out_probs[i], out_labels[i])
        
    results = {"substance_id": out_names, "Probability": out_probs, "Label": out_labels}
    results = pd.DataFrame.from_dict(results, orient="index").T
    results.to_csv("results55.csv", index=None)

#visualization	
visualization(test_substances, test_preds, test_inputs, test_targets, test_atom_activations_vis, test_att_prob_vis, model_params)
