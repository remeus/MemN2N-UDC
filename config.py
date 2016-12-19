""" Main hyperperameters."""
import tensorflow as tf
from collections import namedtuple

# Data Parameters
tf.flags.DEFINE_integer("vocab_size", 54538, "The size of the vocabulary")

# Embedding Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings [100]")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length [160]")
tf.flags.DEFINE_integer("max_utterance_len", 160, "Truncate utterance to this length [160]")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 2e-4, "Learning rate [2e-4 MN, 1e-3 DSSM-LSTM]")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training [128]")
tf.flags.DEFINE_integer("eval_batch_size", 163, "Batch size during evaluation [163]")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name [Adam]")
tf.flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50 MN-DSSM, 10-LSTM]")

FLAGS = tf.flags.FLAGS

Config = namedtuple(
  "Config",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "vocab_size",
    "max_grad_norm"
  ])

def create_config():
  return Config(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    max_grad_norm=FLAGS.max_grad_norm)