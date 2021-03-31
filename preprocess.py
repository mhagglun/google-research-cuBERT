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
import tokenization
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from bert import modeling
from argparse import ArgumentParser


tokenizer = tokenization.FullTokenizer(
    vocab_file='./model/vocab.txt', do_lower_case=True)


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


def extract_features_from_sample(sample, seq_length=512):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    tokens_a = tokenizer.tokenize(sample['text_a'])
    # print(tokens_a)
    tokens_b = None
    if sample['text_b']:
        tokens_b = tokenizer.tokenize(sample['text_b'])

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]_")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]_")
    input_type_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("'[SEP]_'")
        input_type_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

    return {"unique_ids": sample['unique_id'], "method_name": sample['method_name'],
            "tokens": tokens, "input_ids": input_ids, "input_mask": input_mask,
            "input_type_ids": input_type_ids}


def extractor(args):
    files = glob.glob(args.directory + '/**/*.py', recursive=True)
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

        # Only consider methods
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name

                if any([True for s in args.filters if method_name.find(s) != -1]):
                    method_string = astunparse.unparse(node)

                    line = tokenization.convert_to_unicode(method_string)
                    if not line:
                        skipped += 1
                        print(
                            f"Skipped file due to tokenization. Total skips: {skipped}")
                        break
                    line = line.strip()
                    text_a = ""
                    text_b = ""
                    m = re.match(r"^(.*) \|\|\| (.*)$", line)
                    if m is None:
                        text_a = line
                    else:
                        text_a = m.group(1)
                        text_b = m.group(2)

                    examples.append({"unique_id": unique_id, "method_name": method_name,
                                    "text_a": text_a, "text_b": text_b})
                    unique_id += 1

    print(
        f"DONE WITH EXTRACTION. SKIPPED {skipped} PROBLEMATIC FILES IN TOTAL")

    with open(args.output, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
                                'unique_ids', 'method_name', 'tokens', 'input_ids', 'input_mask', 'input_type_ids'])
        writer.writeheader()
        for sample in tqdm(examples, desc=f"Writing results to file {args.output}"):
            features = extract_features_from_sample(sample)
            writer.writerow(features)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", type=str,
                        help="The directory of the projects to extract libraries from", required=True)
    parser.add_argument("-o", "--output", dest="output", type=str, default="./data/preprocessed.csv",
                        help="The output filepath for the csv file", required=False)
    parser.add_argument("-v", "--verbose", dest="verbose", type=bool, default=False,
                        help="Increase verbosity of output", required=False)
    parser.add_argument("-f", "--filters", dest="filters", type=lambda s: [str(item) for item in s.split(',')], default="",
                        help="Filters for method names include during extraction", required=False)

    args = parser.parse_args()
    extractor(args)
    print("Done.")
