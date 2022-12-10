# Modified version of Google's data_utils for processing WMT data
#
# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================

"""Utilities for downloading data from OpenSubtitles, parsing, vocabularies.
Main function is prepare_open_subtitles_data() which downloads the OpenSubtitles
data set to a specified data_dir and creates the following structure
data_dir/
  OpenSubtitles/
    (unzipped contents)
  train/
    train_inputs.txt
    train_targets.txt
    *.xml (transcripts)
  valid/
    valid_inputs.txt
    valid_targets.txt
    *.xml (transcripts)
  vocab100000.en
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import xml.etree.ElementTree
import random
import time
import sys

from tensorflow.python.platform import gfile
from six.moves import urllib


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Used for writing training data files
SENT_SEPARATOR  = '|||'

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

# Used to generate pseudorandom values for train/validation split
random.seed(42)

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

_OPEN_SUBTITLES_DATA_URL = "http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2012/en.tar.gz"

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath, reporthook)
    statinfo = os.stat(filepath)
    print("Succesfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  # print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "w") as new_file:
      for line in gz_file:
        new_file.write(line)


def listFiles(directory):
    rootdir = directory
    for root, subFolders, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(root,f)
    return


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def parse_open_subtitles_xml(xml_path):
  """Get a list of utterances from OpenSubtitles transcript at xml_path
  Returns: a list of lists, where each utterance is a list of tokens
  """
  def encode_utf8(s):
    if s:
      return s.encode('utf-8')
    return ''

  try:
    e = xml.etree.ElementTree.parse(xml_path).getroot()
    utterances = []
    for s in e.findall('s'):
      utterance = [encode_utf8(w.text) for w in s.findall('w')]
      utterances.append(utterance)
    return utterances
  except:
    print("Unable to parse", xml_path)
    print("Unexpected error:", sys.exc_info()[0])
    return []

def add_data_points_to_file(transcript_xml, inputs_file, targets_file):
  """Parses XML of file at location transcript_xml. Each pair of consecutive sentences
  defines a data point, where the i^th data point is defined by the i^th line of inputs_file 
  and i^th line of the targets_file.
  inputs_file and targets_file should be Python file objects, opened in append mode.
  """  
  utterances = parse_open_subtitles_xml(transcript_xml)
  for i in range(len(utterances)-1):
    x = utterances[i]
    y = utterances[i+1]
    inputs_file.write(' '.join(x) + '\n')
    targets_file.write(' '.join(y) + '\n')
  return


def process_open_subtitles_data_set(extracted_dir, data_dir):
  """ Called by get_open_subtitles_data_set()
  Arguments: extracted_dir is the path to the extracted contents of OpenSubtitles data set. Generally this is named OpenSubtitles
  data_dir is the path where the training and validation sets will be written to.
  Returns: a pair, the first element is the training directory path, and the second element is the validation directory path.
  """
  train_dir          = data_dir  + '/train'
  valid_dir          = data_dir  + '/valid'
  train_inputs_path  = train_dir + '/train_inputs.txt'
  train_targets_path = train_dir + '/train_targets.txt'
  valid_inputs_path  = valid_dir + '/valid_inputs.txt'
  valid_targets_path = valid_dir + '/valid_targets.txt'

  # train_dir and valid_dir should either not exist or be empty
  # if (gfile.Exists(train_dir) and len(gfile.ListDirectory(train_dir)) > 0):
  #   response = 'n'
  #   while response[0] != 'y' and response[0] != 'n':
  #     response = raw_input("Warning: Contents of %s will be deleted and overwritten. Do you wish to continue?" +
  #       "y to delete, n to use existing train and validation sets, or press CTRL+C to exit", train_dir)
  #   if response[0] == 'n':
  #     return (train_dir, valid_dir)
  #   gfile.DeleteRecursively(train_dir)

  # if (gfile.Exists(valid_dir) and len(gfile.ListDirectory(valid_dir)) > 0):
  #   response = 'n'
  #   while response[0] != 'y' and response[0] != 'n':
  #     response = raw_input("Warning: Contents of %s will be deleted and overwritten. Do you wish to continue?" +
  #       "y to delete, n to use existing train and validation sets, or press CTRL+C to exit", valid_dir)
  #   if response[0] == 'n':
  #     return (train_dir, valid_dir)
  #   gfile.DeleteRecursively(valid_dir)

  # if not (gfile.IsDirectory(train_dir)):
  #   gfile.MkDir(train_dir)
  # if not (gfile.IsDirectory(valid_dir)):
  #   gfile.MkDir(valid_dir)

  if not (gfile.Exists(train_dir) and len(gfile.ListDirectory(train_dir)) and 
    gfile.Exists(valid_dir) and len(gfile.ListDirectory(valid_dir))):
    print("Writing training data to %s and validation data to %s" % (train_dir, valid_dir))
    if not (gfile.IsDirectory(train_dir)):
      gfile.MkDir(train_dir)
    if not (gfile.IsDirectory(valid_dir)):
      gfile.MkDir(valid_dir)

    train_inputs_file  = open(train_inputs_path, 'w+')
    train_targets_file = open(train_targets_path, 'w+')
    valid_inputs_file  = open(valid_inputs_path, 'w+')
    valid_targets_file = open(valid_targets_path, 'w+')

    counter = 0
    for f in listFiles(extracted_dir):
      if f[-3:] == '.gz':
        # Trim the .gz and unzip to data_dir
        counter += 1
        if counter%1000 == 0:
          print("Processed %d transcripts" % counter)
        if random.random() < .7:
          dest_path = data_dir + '/train/' + f.split('/')[-1][:-3]
          gunzip_file(f, dest_path)
          add_data_points_to_file(dest_path, train_inputs_file, train_targets_file)
        else:
          dest_path = data_dir + '/valid/' + f.split('/')[-1][:-3]
          gunzip_file(f, dest_path)
          add_data_points_to_file(dest_path, valid_inputs_file, valid_targets_file)
  else:
    print("Training and validation directories at %s and %s contain content. \
      Skipping creation of training and validation sets." % (train_dir, valid_dir))
  return (train_dir, valid_dir)


def get_open_subtitles_data_set(data_dir):
  """Download the WMT en-fr training corpus to directory unless it's there or data_dir already present.
  """
  file_path = os.path.join(data_dir, "open_subtitles_en.tar.gz")
  if not (gfile.Exists(file_path)):
    corpus_file = maybe_download(data_dir, "open_subtitles_en.tar.gz",
                                 _OPEN_SUBTITLES_DATA_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      # Extracts everything to a folder called OpenSubtitles. (assumption)
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(corpus_tar, data_dir)

  return process_open_subtitles_data_set(data_dir + '/OpenSubtitles2012', data_dir)


def create_vocabulary(vocabulary_path, data_dir, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_dir: data directory that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data in directory %s" % (vocabulary_path, data_dir))
    vocab = {}
    counter = 0
    for data_file in gfile.ListDirectory(data_dir):
      # Assume all xml files are data files, and vice versa
      if data_file[-3:] != 'xml':
        continue
      counter += 1
      if counter%10000 == 0:
        print("Processing file %d" % counter)
      utterances = parse_open_subtitles_xml(data_dir + '/' + data_file)
      for utterance in utterances:
        for w in utterance:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:	
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_open_subtitles_data(data_dir, vocabulary_size):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the vocabulary to create and use.

  Returns:
    A tuple of 5 elements:
      (1) path to the token-ids for training input data-set,
      (2) path to the token-ids for training output data-set,
      (3) path to the token-ids for validation input data-set,
      (4) path to the token-ids for validation output data-set,
      (5) path to the vocabulary file,
  """
  # Get wmt data to the specified directory.
  (train_dir, valid_dir) = get_open_subtitles_data_set(data_dir)

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab%d.en" % vocabulary_size)
  create_vocabulary(vocab_path, train_dir, vocabulary_size)

  # Create token ids for the training data.
  train_inputs_ids_path = data_dir + ("/train.ids%d.in" % vocabulary_size)
  data_to_token_ids(train_dir + "/train_inputs.txt", train_inputs_ids_path, vocab_path)
  train_targets_ids_path = data_dir + ("/train.ids%d.out" % vocabulary_size)
  data_to_token_ids(train_dir + "/train_targets.txt", train_targets_ids_path, vocab_path)

  # Create token ids for the development data.
  valid_inputs_ids_path = data_dir + ("/valid.ids%d.in" % vocabulary_size)
  data_to_token_ids(valid_dir + "/valid_inputs.txt", valid_inputs_ids_path, vocab_path)
  valid_targets_ids_path = data_dir + ("/valid.ids%d.out" % vocabulary_size)
  data_to_token_ids(valid_dir + "/valid_targets.txt", valid_targets_ids_path, vocab_path)

  return (train_inputs_ids_path, train_targets_ids_path, 
    valid_inputs_ids_path, valid_targets_ids_path,
    vocab_path)
