from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import codecs
import collections
import json
import re
import ast
import astunparse
import glob
import csv

import itertools
import tokenizer_registry
import code_to_subtokenized_sentences

from absl import app
from absl import flags
from tensor2tensor.data_generators import text_encoder

import tensorflow as tf
import numpy as np

from tqdm import tqdm
from bert import modeling
from python_tokenizer import PythonTokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string('vocabulary_filepath', None,
                    'Path to the subword vocabulary.')

flags.DEFINE_string('input_filepath', None,
                    'Path to the Python source code file.')

flags.DEFINE_string('output_filepath', None,
                    'Path to the output file of subtokenized source code.')

flags.DEFINE_enum_class(
    'tokenizer',
    default=tokenizer_registry.TokenizerEnum.PYTHON,
    enum_class=tokenizer_registry.TokenizerEnum,
    help='The tokenizer to use.')

def tokenize(sample, tokenizer, subword_tokenizer, seq_length=512):
    
    code = sample['method_body'] 

    subtokenized_sentences, sentence_ids = (
        code_to_subtokenized_sentences.code_to_cubert_sentences(
            code=code,
            initial_tokenizer=tokenizer,
            subword_tokenizer=subword_tokenizer))

    # iterate over each sentence
    tokenized_code = []
    tokenized_code.append(['[CLS]_'])

    input_ids = []
    input_ids.append([2])

    for sentence, sentence_id in zip(subtokenized_sentences, sentence_ids):
        tokenized_code.append(sentence)
        tokenized_code.append(['[SEP]_'])

        input_ids.append(sentence_id)
        input_ids.append([3])
    
    # Truncate code if too long
    while True:
        num_tokens = sum([len(s) for s in tokenized_code])
        if num_tokens <= seq_length:
            break
        else:
            tokenized_code.pop()
            input_ids.pop()

    # unpack list of sentences into a single list
    tokenized_code = list(itertools.chain(*tokenized_code))
    input_ids = list(itertools.chain(*input_ids))

    input_type_ids = [0] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return {"unique_ids": sample['unique_id'], "filepath": sample['filepath'], "label": sample['label'], "method_name": sample['method_name'],
            "tokens": tokenized_code, "input_ids": input_ids, "input_mask": input_mask, "input_type_ids": input_type_ids}

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    tokenizer = FLAGS.tokenizer.value()
    subword_tokenizer = text_encoder.SubwordTextEncoder(
        FLAGS.vocabulary_filepath)

    files = glob.glob(FLAGS.input_filepath + '/**/*.py', recursive=True)
    examples = []
    unique_id = 0
    skipped = 0
    for fname in tqdm(files, desc="Extracting methods"):
        try:
            with open(fname) as fh:
                root = ast.parse(fh.read(), fname)
        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"Skipping problematic file {e}", fname, file=sys.stderr)
            continue
        
        # Get label from parent folder of file
        label = fname.split('/')[5]

        # Only consider methods
        for node in ast.iter_child_nodes(root):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name
                method_string = astunparse.unparse(node)
                method_body = astunparse.unparse(node.body)
                examples.append({"unique_id": unique_id, "filepath": fname, "label": label,
                                "method_name": method_name, "method_string": method_string, "method_body": method_body})
                unique_id += 1

    print(
        f"DONE WITH EXTRACTION. SKIPPED {skipped} PROBLEMATIC FILES IN TOTAL")

    with open(FLAGS.output_filepath, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                'unique_ids', 'filepath', 'label', 'method_name', 'tokens', 'input_ids', 'input_mask', 'input_type_ids'])
        writer.writeheader()
        for sample in tqdm(examples, desc=f"Writing results to file {FLAGS.output_filepath}"):
            tokens = tokenize(sample, tokenizer, subword_tokenizer)
            writer.writerow(tokens)


if __name__ == '__main__':
    flags.mark_flag_as_required('vocabulary_filepath')
    flags.mark_flag_as_required('input_filepath')
    flags.mark_flag_as_required('output_filepath')
    app.run(main)