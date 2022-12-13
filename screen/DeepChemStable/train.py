
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
from data_features import *
from model import *
from utils import *
import tensorflow as tf

# parameters
task_params = dict(target_name = "label",
                   #data_file = "data/Stable_Unstable_shuffled9741.csv",
                   N_train = 7792, 
                   N_val = 974, 
                   N_test = 975)

train_params = dict(
                    n_batches = 500,
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

traindata, valdata, testdata = load_data(
    task_params['data_file'], (task_params["N_train"], task_params["N_val"], task_params["N_test"]),
    input_name='smiles', target_name=task_params['target_name'])
train_inputs, train_labels = traindata
val_inputs,   val_labels   = valdata
test_inputs,  test_labels = testdata

train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

train_labels = tf.reshape(train_labels, [-1, 1])
val_labels = tf.reshape(val_labels, [-1, 1])
test_labels = tf.reshape(test_labels, [-1, 1])

train_substances = array_rep_from_smiles(tuple(train_inputs))
val_substances = array_rep_from_smiles(tuple(val_inputs))
test_substances = array_rep_from_smiles(tuple(test_inputs))

train_substances =  trans(train_substances)
val_substances =  trans(val_substances)
test_substances =  trans(test_substances)

fingerprint_variables = initialize_fingerprint_variables(train_params, model_params)
prediction_variables = initialize_convolution_prediction_variables(train_params, model_params)

# train
train_fps, train_rnn_embedding, train_atom_activations = build_neural_fps_network(train_substances, fingerprint_variables, model_params)
train_fps, train_att_prob = build_attention(train_rnn_embedding, train_substances, model_params, is_training=True)
train_normed_predictions = build_convolution_prediction_network(train_fps,
                                                                prediction_variables, model_params)
train_predictions, train_loss = build_loss_network(train_normed_predictions, 
                                                   train_labels, fingerprint_variables, prediction_variables, model_params)
optimizer = build_optimizer(train_loss, train_params)


# val
val_fps, val_rnn_embedding, val_atom_activations= build_neural_fps_network(val_substances, fingerprint_variables, model_params)
val_fps, val_att_prob = build_attention(val_rnn_embedding, val_substances, model_params, is_training=False)
val_normed_predictions = build_convolution_prediction_network(val_fps,
                                                                prediction_variables, model_params)
val_predictions, val_loss = build_loss_network(val_normed_predictions, 
                                                   val_labels, fingerprint_variables, prediction_variables, model_params)


# test
test_fps, test_rnn_embedding,  test_atom_activations = build_neural_fps_network(test_substances, fingerprint_variables, model_params)
test_fps, test_att_prob = build_attention(test_rnn_embedding, test_substances, model_params, is_training=False)
test_normed_predictions = build_convolution_prediction_network(test_fps,
                                                                prediction_variables, model_params)
test_predictions, test_loss = build_loss_network(test_normed_predictions, 
                                                   test_labels, fingerprint_variables, prediction_variables, model_params)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    best_step = 0
    best_val_accuracy = 0
    for train_step in range(train_params['n_batches']):
        _, loss, train_preds, train_targets,  train_atom_activations_vis, train_att_prob_vis = sess.run(fetches=[optimizer,
                                                                                                        train_loss, train_predictions, train_labels, 
                                                                                                        train_atom_activations, train_att_prob])
        val_preds, val_targets = sess.run(fetches=[val_predictions, val_labels])
        
        train_accuracy = accuracy(train_preds, train_targets)
        val_accuracy =  accuracy(val_preds, val_targets)
        val_auc = auc(val_preds, val_targets)
        val_recall = FNR(val_preds, val_targets)
        
        print("train_step", train_step, "loss:", loss, "train_accuracy: ", train_accuracy)
        print("*******", "val_accuracy: ", val_accuracy,  
              "val_auc", val_auc,"val_recall", val_recall)
 
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_step = train_step
            
        if train_step == 498:
            fingerprint_variables_out = {}
            prediction_variables_out = {}
            for key in fingerprint_variables.keys():
                if  key == "n_params":
                    fingerprint_variables_out[key] = fingerprint_variables[key]
                else:
                    fingerprint_variables_out[key] = sess.run(fingerprint_variables[key])
            for key in prediction_variables.keys():
                if  key == "n_params":
                    prediction_variables_out[key] = prediction_variables[key]
                else:
                    prediction_variables_out[key] = sess.run(prediction_variables[key])
                
    print("best step: ", best_step, "best_val_accuracy", best_val_accuracy)

#  output trained variables
import pickle
f=open('./DeepChemStable_model/fingerprint_variables.json','wb')
pickle.dump(fingerprint_variables_out, f)
f.close()

f=open('./DeepChemStable_model/prediction_variables.json','wb')
pickle.dump(prediction_variables_out, f)
f.close()

#visualization
visualization(val_substances, val_pred, val_inputs, val_targets, val_atom_activations_vis, val_att_prob_vis, model_params)

