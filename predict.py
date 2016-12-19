"""Predict a response given a context and a set of utterances"""
import sys
from random import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
import model
import config
from models.mn.model import mn
from models.lstm.model import lstm
from models.dssm.model import dssm

tf.flags.DEFINE_string("model_dir", "checkpoints", "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "data/vocab_processor.bin", "Saved vocabulary processor file")
tf.flags.DEFINE_string("network", "MN",
                       "Type of model (MN for Memory Network, LSTM for Long-Short Term Memory, "
                       "DSSM for Deep Structured Semantic Model [MN]")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

# Select model
if FLAGS.network == "LSTM":
  selected_model = lstm
elif FLAGS.network == "DSSM":
  selected_model = dssm
else:
  selected_model = mn

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Load vocabulary
vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
  FLAGS.vocab_processor_file)

# Load random row
val = pd.read_csv('data/valid.csv', sep=',', header=0)
ind = np.random.randint(len(val))
INPUT_CONTEXT = val['Context'].values[ind]
POTENTIAL_RESPONSES = [val['Ground Truth Utterance'].values[ind]]
for i in range(9):
    POTENTIAL_RESPONSES.append(val['Distractor_{}'.format(i)].values[ind])


def get_features(context, utterance):
  """Build features"""
  context_matrix = np.array(list(vp.transform([context])))
  context_len = len(context.split(" "))
  utterance_matrix = np.array(list(vp.transform([utterance[0]])))
  utterance_len = len(utterance[0].split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1, 1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1, 1], dtype=tf.int64),
  }
  for i in range(1, 10):
    features["distractor_{}".format(i-1)] = tf.convert_to_tensor(np.array(list(vp.transform([utterance[i]]))), dtype=tf.int64)
    features["distractor_{}_len".format(i-1)] = tf.constant(len(utterance[i].split(" ")), shape=[1, 1], dtype=tf.int64)
  return features, None


if __name__ == "__main__":

  # Load hyperparameters
  hparams = config.create_config()

  # Prepare function that will be used for loading context/utterance
  model_fn = model.create_model_fn(hparams, model_impl=selected_model)

  # Load estimator
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)
  estimator._targets_info = tf.contrib.learn.estimators.tensor_signature.TensorSignature(tf.constant(0, shape=[1,1]))

  shuffle(POTENTIAL_RESPONSES)

  # Compute prediction
  print("Context: {}".format(INPUT_CONTEXT))
  prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, POTENTIAL_RESPONSES))
  for r in range(len(POTENTIAL_RESPONSES)):
    print("{}: {:g}".format(POTENTIAL_RESPONSES[r], prob[0, r]))

