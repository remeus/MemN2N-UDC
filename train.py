# Copyrights & references
#
# This code is built using results from a few papers:
# --
# The Memory Network architecture is based on End-To-End Memory Networks by Sainbayar Sukhbaatar, Arthur Szlam,
# Jason Weston, Rob Fergus (https://arxiv.org/abs/1503.08895v4)
# --
# The Ubuntu Dialog Corpus and the dual encoder model are both described in The Ubuntu Dialogue Corpus: A Large
# Dataset for Research in Unstructured Multi-Turn Dialogue Systems by Ryan Lowe, Nissan Pow, Iulian Serban,
# Joelle Pineau (https://arxiv.org/abs/1506.08909)
#
# Three practical implementations have been used for this project:
# --
# Dual LSTM Encoder for Dialog Response Generation by Denny Britz (https://github.com/dennybritz/chatbot-retrieval)
# --
# End-To-End Memory Network using Tensorflow by Dominique Luna (https://github.com/domluna/memn2n)
# --
# A Tensorflow implementation of DSSM (slightly modified) (https://github.com/ShuaiyiLiu/sent_cnn_tf)
# ================================================================================================

"""Training Ops."""
import os
import time
import shutil

import tensorflow as tf

import model
import config
import metrics
import inputs
from models.mn.model import mn
from models.lstm.model import lstm
from models.dssm.model import dssm

tf.flags.DEFINE_string("input_dir", "data",
                       "Directory containing input data files 'train.tfrecords' and 'validation.tfrecords'")
tf.flags.DEFINE_string("model_dir", "checkpoints", "Directory to store model checkpoints (defaults to ./runs)")
tf.flags.DEFINE_integer("num_epochs", None, "Number of training Epochs. Defaults to indefinite.")
tf.flags.DEFINE_integer("eval_every", 1000, "Evaluate after this many train steps")
tf.flags.DEFINE_string("network", "MN",
                       "Type of model (MN for Memory Network, LSTM for Long-Short Term Memory, "
                       "DSSM for Deep Structured Semantic Model [MN]")
FLAGS = tf.flags.FLAGS

# Output directory
TIMESTAMP = int(time.time())
if os.path.exists(FLAGS.model_dir):
    shutil.rmtree(FLAGS.model_dir)
if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    MODEL_DIR = os.path.abspath(os.path.join("./runs", str(TIMESTAMP)))

# Input files
TRAIN_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "train.tfrecords"))
VALIDATION_FILE = os.path.abspath(os.path.join(FLAGS.input_dir, "validation.tfrecords"))

# Select model
if FLAGS.network == "LSTM":
    selected_model = lstm
elif FLAGS.network == "DSSM":
    selected_model = dssm
else:
    selected_model = mn

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    """Launch training"""

    # Load hyperparameters
    params = config.create_config()

    # Prepare function that will be used for loading context/utterance
    model_fn = model.create_model_fn(
        params,
        model_impl=selected_model)

    # Prepare estimator
    estimator = tf.contrib.learn.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        config=tf.contrib.learn.RunConfig(gpu_memory_fraction=0.25,
                                          save_checkpoints_secs=60 * 2,
                                          keep_checkpoint_max=1,
                                          log_device_placement=False))

    # Prepare input training examples
    input_fn_train = inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.TRAIN,
        input_files=[TRAIN_FILE],
        batch_size=params.batch_size,
        num_epochs=FLAGS.num_epochs,
        params=params)

    # Prepare input validation examples
    input_fn_eval = inputs.create_input_fn(
        mode=tf.contrib.learn.ModeKeys.EVAL,
        input_files=[VALIDATION_FILE],
        batch_size=params.eval_batch_size,
        num_epochs=1,
        params=params)

    # Load recall metrics for validation
    eval_metrics = metrics.create_evaluation_metrics()

    # Prepare monitor for validation
    eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_eval,
        every_n_steps=FLAGS.eval_every,
        metrics=eval_metrics)

    # Lauch training
    estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])


if __name__ == "__main__":
    tf.app.run()
