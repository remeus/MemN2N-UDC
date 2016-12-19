"""Test the model over the testing dataset"""
import sys

import tensorflow as tf

import config
import model
import metrics
import inputs
from models.mn.model import mn
from models.lstm.model import lstm
from models.dssm.model import dssm

tf.flags.DEFINE_string("test_file", "data/test.tfrecords", "Path of test data in TFRecords format")
tf.flags.DEFINE_string("model_dir", "checkpoints", "Directory to load model checkpoints from")
tf.flags.DEFINE_integer("test_batch_size", 110, "Batch size for testing [110]")
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

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == "__main__":

  # Load hyperparameters
  hparams = config.create_config()

  # Prepare function that will be used for loading context/utterance
  model_fn = model.create_model_fn(hparams, model_impl=selected_model)

  # Prepare estimator
  estimator = tf.contrib.learn.Estimator(
    model_fn=model_fn,
    model_dir=FLAGS.model_dir,
    config=tf.contrib.learn.RunConfig(gpu_memory_fraction=0.25))

  # Prepare input testing examples
  input_fn_test = inputs.create_input_fn(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    input_files=[FLAGS.test_file],
    batch_size=FLAGS.test_batch_size,
    num_epochs=1,
    params=hparams)

  eval_metrics = metrics.create_evaluation_metrics()

  # Lauch testing
  estimator.evaluate(input_fn=input_fn_test, steps=None, metrics=eval_metrics)
