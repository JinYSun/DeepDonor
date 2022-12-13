from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
tf.random.set_seed(321)
from data_features import num_atom_features, num_bond_features

degrees = [1, 2, 3, 4]

def _add_weights(variables, weight_key, shape, train_params, op=tf.random.normal):
    weights = tf.Variable(
        initial_value=op(shape, stddev=np.exp(train_params['log_init_scale'])),
        name=weight_key)
    variables[weight_key] = weights
    with tf.name_scope("regularization/") as regularization_scope:
        variables['l2_loss'] += tf.nn.l2_loss(weights)
        variables['n_params'] += np.prod(shape)

def initialize_fingerprint_variables(train_params, model_params):
    variables = {}
    with tf.name_scope("regularization/") as scope:
        variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
        variables['n_params'] = 0

    with tf.name_scope("neural_fingerprint") as scope:
        all_layer_sizes = [num_atom_features()] + [model_params['fp_width']] * model_params['fp_depth']
        for layer in range(len(all_layer_sizes)):
            with tf.name_scope("layer_{}".format(layer)):
                _add_weights(
                    variables = variables,
                    weight_key = 'layer_output_weights_{}'.format(layer),
                    shape = [all_layer_sizes[layer], model_params['fp_length']],
                    train_params = train_params)

                _add_weights(
                    variables = variables,
                    weight_key = 'layer_output_bias_{}'.format(layer),
                    shape = [model_params['fp_length']],
                    train_params = train_params)

            in_and_out_sizes = zip(all_layer_sizes[:-1], all_layer_sizes[1:])
            for layer, (N_prev, N_cur) in enumerate(in_and_out_sizes):
                with tf.name_scope("layer_{}/".format(layer)) as layer_scope:
                    _add_weights(
                        variables = variables,
                        weight_key = "layer_{}_biases".format(layer),
                        shape = [N_cur],
                        train_params = train_params)

                    _add_weights(
                        variables = variables,
                        weight_key = "layer_{}_self_filter".format(layer),
                        shape = [N_prev, N_cur],
                        train_params = train_params)

                    for degree in degrees:
                        _add_weights(
                            variables = variables,
                            weight_key = 'layer_{}_neighbor_{}_filter'.format(layer, degree),
                            shape = [N_prev + num_bond_features(), N_cur],
                            train_params = train_params)
    return variables

def initialize_convolution_prediction_variables(train_params, model_params):
    variables = {}
    with tf.name_scope("regularization/") as scope:
        variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
        variables['n_params'] = 0

    with tf.name_scope("convolution_prediction") as scope:
        layer_sizes = model_params['prediction_layer_sizes'] + [1]
        for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            _add_weights(
                variables = variables,
                weight_key = "prediction_weights_{}".format(i),
                shape = shape,
                train_params = train_params)

            _add_weights(
                variables = variables,
                weight_key = "prediction_biases_{}".format(i),
                shape = [shape[1]],
                train_params = train_params)

    return variables

def initialize_linear_regression_prediction_variables(train_params, model_params):
    variables = {}
    with tf.name_scope("regularization/") as scope:
        variables['l2_loss'] = tf.constant(0.0, name="l2_loss")
        variables['n_params'] = 0

    with tf.name_scope("linear_regression_prediction") as scope:
        _add_weights(
            variables = variables,
            weight_key = "linear_regression_weights".format(i),
            shape = [model_params["fp_length"]],
            train_params = train_params)

    return variables

def build_neural_fps_network(substances, variables, model_params):

    def matmult_neighbors(atom_features, layer, substances, variables):
 
        with tf.name_scope("matmul_neighbors/") as matmul_neighbors_scope:
            activations_by_degree = []
            for degree in degrees:
                atom_neighbor_list = substances['atom_neighbors_{}'.format(degree)]
                bond_neighbor_list = substances['bond_neighbors_{}'.format(degree)]
                neighbor_features = [
                    tf.gather(params=atom_features, indices=atom_neighbor_list),
                    tf.gather(params=substances['bond_features'], indices=bond_neighbor_list)]
                stacked_neighbors = tf.concat(axis=2, values=neighbor_features)
                summed_neighbors = tf.reduce_sum(stacked_neighbors, axis=1)
                neighbor_filter = variables['layer_{}_neighbor_{}_filter'.format(layer, degree)]
                activations = tf.matmul(summed_neighbors, neighbor_filter)
                activations_by_degree.append(activations)
                activations = tf.concat(
                    axis=0, values=activations_by_degree, name="activations")
            return activations

    def update_layer(atom_features, layer, substances, variables):
        with tf.name_scope("layer_{}/".format(layer)) as update_layer_scope:
            layer_bias = variables["layer_{}_biases".format(layer)]
            layer_self_filter = variables["layer_{}_self_filter".format(layer)]
            self_activations = tf.matmul(atom_features, layer_self_filter)
            neighbor_activations = matmult_neighbors(
                atom_features, layer, substances, variables)
            
            activations = tf.nn.bias_add(self_activations, layer_bias)
            activations_mean, activations_variance = tf.nn.moments(activations, [0], keepdims=True)
            # batch normalization a la neural fingerprints
            activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)

            #activations = tf.nn.batch_normalization(
            #activations, activations_mean, activations_variance,
            #offset=None, scale=None, variance_epsilon=1e-3)
            activations = tf.nn.relu(activations, name="activations")
            return activations

    atom_activations = []
    def write_to_fingerprint(atom_features, layer, substances, variables):
        
        with tf.name_scope("layer_{}/".format(layer)) as scope:
            out_weights = variables['layer_output_weights_{}'.format(layer)]
            out_bias = variables['layer_output_bias_{}'.format(layer)]
            hidden = tf.nn.bias_add(tf.matmul(atom_features, out_weights), out_bias)
            atom_outputs = tf.nn.softmax(hidden)
            layer_output = tf.sparse.sparse_dense_matmul(
                substances['substance_atoms'], atom_outputs, name=scope)
            
            # for visualization
            atom_activations.append(atom_outputs)
            
            # temp for check
            #atom_activations.append(out_weights)
            
            return layer_output, atom_outputs

    with tf.name_scope("fingerprint/") as fingerprint_scope:
        atom_features = substances['atom_features']
        fps, rnn_embedding = write_to_fingerprint(atom_features, 0, substances, variables)
        
        num_hidden_features = [model_params['fp_width']] * model_params['fp_depth']
        for layer in xrange(len(num_hidden_features)):
            atom_features = update_layer(atom_features, layer, substances, variables)
            fps_layer, rnn_embedding = write_to_fingerprint(atom_features, layer+1, substances, variables)
            fps += fps_layer
            rnn_embedding += rnn_embedding
        return fps, rnn_embedding, atom_activations

def build_attention(rnn_embedding, substances, model_params, is_training=False):
        last_add = tf.constant(0, shape=[1, model_params["fp_length"]], dtype=tf.float32)
        rnn_embedding = tf.concat([rnn_embedding, last_add], axis=0)

        embedded_seq = tf.nn.embedding_lookup(rnn_embedding, substances["rnn_raw_input"])
        q = embedded_seq
        kT = tf.transpose(q, perm=[0, 2, 1])
        qkT = tf.matmul(q, kT)
        qkT = tf.divide(qkT, tf.sqrt(tf.cast(kT.shape[2], dtype=tf.float32)))
        att_prob = tf.nn.softmax(qkT)
        att = tf.matmul(att_prob, q)
        print("attention atoms shape", att)
        att_fps = tf.reduce_sum(att, axis=1)
        
        return att_fps, att_prob
    
def build_convolution_prediction_network(fps, variables, model_params):

    with tf.name_scope("convolution_prediction") as convolution_prediction_scope:
        activations = fps
        layer_sizes = model_params['prediction_layer_sizes'] + [1]
        for layer in range(len(layer_sizes) - 1):
            weights = variables['prediction_weights_{}'.format(layer)]
            biases = variables['prediction_biases_{}'.format(layer)]
            activations = tf.nn.bias_add(tf.matmul(activations, weights), biases, name="activations")
            if layer < len(layer_sizes) - 2:
                activations_mean, activations_variance = tf.nn.moments(activations, [0], keepdims=True)
                activations = (activations - activations_mean) / (tf.sqrt(activations_variance) + 1)
                activations = tf.nn.relu(activations)
        return tf.squeeze(activations, name=convolution_prediction_scope)

def build_linear_prediction_network(fps, variables, model_params):

    with tf.name_scope("linear_prediction") as scope:
        weights = variables['linear_prediction_weights']
        return tf.mul(fps, weights, name=scope)

def build_loss_network(
    normed_predictions,
    labels,
    fingerprint_variables,
    prediction_variables,
    model_params):

    with tf.name_scope("loss") as loss_scope:

        normed_predictions = tf.reshape(normed_predictions, [-1, 1])
        scaled = 1 / tf.reduce_sum(tf.reduce_sum(model_params["pos_weight"]*labels) + tf.reduce_sum(1*(1-labels)))
        cross_entropy = scaled * tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(labels=labels, logits=normed_predictions, pos_weight=model_params["pos_weight"]))

        fingerprint_regularization = fingerprint_variables['l2_loss'] * model_params['l2_penalty'] / fingerprint_variables['n_params']
        prediction_regularization = prediction_variables['l2_loss'] * model_params['l2_penalty'] / prediction_variables['n_params']
        regularization = tf.add(
            fingerprint_regularization, prediction_regularization,
            name="regularization")

        loss = tf.add(cross_entropy, regularization, name=loss_scope)
        return normed_predictions, loss

def build_optimizer(loss, train_params):
    with tf.name_scope("optimizer") as optimizer_scope:
        learning_rate = tf.constant(np.exp(train_params['log_learning_rate']))
        beta1 = tf.constant(np.exp(train_params['log_b1']))
        beta2 = tf.constant(np.exp(train_params['log_b2']))
        adam = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
        optimizer = adam.minimize(loss, name=optimizer_scope)
        return optimizer
