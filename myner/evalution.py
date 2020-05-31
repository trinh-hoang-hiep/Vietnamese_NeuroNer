import tensorflow as tf
import spacy
nlp = spacy.load('vi_spacy_model')
# from __future__ import print_function
import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from distutils import util
from neuroner import neuromodel
# fname='model_00017.ckpt'
# model=tf.keras.models.load_model(fname)
# model.predict("3 giờ")


# # build your model (same as training)
# init_op = tf.initialize_all_variables()
# ### Here Comes the fake variable that makes defining a saver object possible.
# _ = tf.Variable(initial_value='fake_variable')
# # sess = tf.Session()
# # saver = tf.train.Saver()
# # saver.restore(sess, 'model_00017.ckpt')
# # session.run(y_pred, feed_dict={x: '3 giờ'})


# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('model_00027.ckpt.meta')
#     saver.restore(sess, "model_00027.ckpt")


# # Recreate the EXACT SAME variables
# v1 = tf.Variable(..., name="v1")
# v2 = tf.Variable(..., name="v2")
# # Now load the checkpoint variable values
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     saver.restore(sess, "model_00027.ckpt")

arguments={
        "character_embedding_dimension": 25,
        "character_lstm_hidden_state_dimension": 25,
        "check_for_digits_replaced_with_zeros": 1,
        "check_for_lowercase": 1,
        "dataset_text_folder": "../neuroner/data/du_doan",
        "debug": 0,
        "dropout_rate": 0.5,
        "experiment_name": "test",
        "fetch_data": "",
        "fetch_trained_model": "",
        "freeze_token_embeddings": 0,
        "gradient_clipping_value": 5.0,
        "learning_rate": 0.05,
        "load_all_pretrained_token_embeddings": 0,
        "load_only_pretrained_token_embeddings": 0,
        "main_evaluation_mode": "conll",
        "maximum_number_of_epochs": 120,
        "number_of_cpu_threads": 8,
        "number_of_gpus": 0,
        "optimizer": "sgd",
        "output_folder": "../myner/output",
        "output_scores": 0,
        "parameters_filepath": "./parameters.ini",
        "patience": 100,
        "plot_format": "pdf",
        "pretrained_model_folder": "../neuroner/trained_models/event",
        "reload_character_embeddings": 1,
        "reload_character_lstm": 1,
        "reload_crf": 1,
        "reload_feedforward": 1,
        "reload_token_embeddings": 1,
        "reload_token_lstm": 1,
        "remap_unknown_tokens_to_unk": 1,
        "spacylanguage": "vi_spacy_model",
        "tagging_format": "bioes",
        "token_embedding_dimension": 300,
        "token_lstm_hidden_state_dimension": 300,
        "token_pretrained_embedding_filepath": "../data/word_vectors/glove.6B.100d.txt",
        "tokenizer": "spacy",
        "train_model": 0,
        "use_character_lstm": 1,
        "use_crf": 1,
        "use_pretrained_model": 1,
        "verbose": 0
    }

from neuroner import neuromodel
nn = neuromodel.NeuroNER(**arguments)
# nn.load("model.ckpt")
# nn.predict(text="Xin chào Bách Khoa lúc 11 giờ trưa") tính năng này k có


# graph = tf.Graph()
# with restored_graph.as_default():
#     with tf.Session() as sess:
#         tf.saved_model.loader.load(
#             sess,
#             [tag_constants.SERVING],
#             'path/to/your/location/',
#         )
#         batch_size_placeholder = graph.get_tensor_by_name('batch_size_placeholder:0')
#         features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
#         labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
#         prediction = restored_graph.get_tensor_by_name('dense/BiasAdd:0')

#         sess.run(prediction, feed_dict={
#             batch_size_placeholder: some_value,
#             features_placeholder: some_other_value,
#             labels_placeholder: another_value
#         })