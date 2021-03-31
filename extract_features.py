# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import sys
import collections
import json
import jsonlines
import re
import ast
import astunparse
import glob
from tqdm import tqdm
from bert import modeling
import tokenization
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "")

flags.DEFINE_string("output_file", None, "")

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("batch_size", 32, "Batch size for predictions.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("master", None,
                    "If using a TPU, the address of the master.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "use_one_hot_embeddings", False,
    "If True, tf.one_hot will be used for embedding lookups, otherwise "
    "tf.nn.embedding_lookup will be used. On TPUs, this should be True "
    "since it is much faster.")


class InputExample(object):

    def __init__(self, unique_id, method_name, text_a, text_b):
        self.unique_id = unique_id
        self.method_name = method_name
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, method_name, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.method_name = method_name
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def input_fn_builder(examples, seq_length, tokenizer):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def generator():
        # Open json, read item by item and yield results
        with open('./data/preprocessed.csv', "r") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=[
                'unique_ids', 'method_name', 'tokens', 'input_ids', 'input_mask', 'input_type_ids'])
            next(reader)
            for row in reader:
                method_name = row['method_name']
                input_ids = ast.literal_eval(row['input_ids'])
                input_mask = ast.literal_eval(row['input_mask'])
                input_type_ids = ast.literal_eval(
                    row['input_type_ids'])
                features = {"unique_ids": int(row['unique_ids']), "method_name": method_name,
                            "input_ids": input_ids, "input_mask": input_mask,
                            "input_type_ids": input_type_ids}
                yield(features)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_generator(generator, output_types={"unique_ids": tf.int32, "method_name": tf.string, "input_ids": tf.int32, "input_mask": tf.int32, "input_type_ids": tf.int32},
                                           output_shapes={"unique_ids": tf.TensorShape([]), "method_name": tf.TensorShape([]), "input_ids": tf.TensorShape([seq_length]), "input_mask": tf.TensorShape([seq_length]), "input_type_ids": tf.TensorShape([seq_length])})
        # d = d.map(extract_features_from_sample)
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d

    return input_fn


def model_fn_builder(bert_config, init_checkpoint, layer_indices, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        scaffold_fn = None
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
             tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indices):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_samples(directory):
    examples = []
    with open('./data/preprocessed.csv', "r") as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=[
            'unique_ids', 'method_name', 'tokens', 'input_ids', 'input_mask', 'input_type_ids'])
        next(reader)
        for row in tqdm(reader, desc="Reading data"):
            method_name = row['method_name']
            tokens = ast.literal_eval(row['tokens'])
            features = {"unique_id": int(row['unique_ids']), "method_name": method_name,
                        "tokens": tokens}
            examples.append(features)
        csvfile.close()
    return examples


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    layer_indices = [int(x) for x in FLAGS.layers.split(",")]

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=FLAGS.master,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    examples = read_samples(FLAGS.input_file)

    unique_id_to_feature = {}
    for ex in examples:
        unique_id_to_feature[ex['unique_id']] = ex

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        layer_indices=layer_indices,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        predict_batch_size=FLAGS.batch_size)

    input_fn = input_fn_builder(
        examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=tokenizer)

    with open(FLAGS.output_file, mode='w') as writer:
        for result in tqdm(estimator.predict(input_fn, yield_single_examples=True), desc="Extracting Features", total=len(examples)):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            output_json = {}
            output_json["unique_id"] = unique_id
            output_json["method_name"] = feature['method_name']

            # feats = {}
            # # Get the features for the CLS token from each layer
            # for (j, layer_index) in enumerate(layer_indices):
            #     layer_output = result["layer_output_%d" % j]
            #     feature = layer_output[0:1]     # CLS token is the first one for each input method
            #     feats["layer_output_%d" % j] = feature

            layer_output = result["layer_output_0"]
            features = layer_output[0:1].tolist()[0]
            # output_json["features"] = features.tolist()[0]
            # writer.write(json.dumps(output_json))
            writer.write(json.dumps({"unique_id": unique_id, "method_name": feature['method_name'], "features": features}) + "\n")


if __name__ == "__main__":
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
