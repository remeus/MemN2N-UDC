"""Manage the MN"""
import tensorflow as tf
from models.mn.mn import MemN2N

FLAGS = tf.flags.FLAGS

def _activation_summary(x, name_tensor=None):
    """Helper to create summaries for activations."""
    if name_tensor is None:
        name_tensor = x.op.name
    tf.histogram_summary(name_tensor + '/activations', x)
    tf.scalar_summary(name_tensor + '/sparsity', tf.nn.zero_fraction(x))


def mn(hparams, mode, context, context_len, utterance, utterance_len, targets):

  if mode == tf.contrib.learn.ModeKeys.INFER:
      batch_size = utterance.get_shape().as_list()[0]
  else:
    batch_size = targets.get_shape().as_list()[0]

  # Build the MN model
  model = MemN2N(hparams.vocab_size, batch_size, hparams.max_context_len, 10, hparams.embedding_dim, hops=3,
                 nonlin=None, initializer=tf.random_normal_initializer(stddev=0.1), encode_position=False, glove=True)

  # Get the output
  output = model.inference(utterance, context)
  _activation_summary(output, "output")

  # Get the probabilities
  probs = tf.nn.softmax(output, name="probs")
  _activation_summary(probs)

  if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None
  else:
      # Display monitoring of main data
      probs_shaped = tf.reshape(probs * 255, [1, -1, 10, 1])
      tf.image_summary('probs', probs_shaped, 20)
      output_shaped = tf.reshape(output, [1, -1, 10, 1])
      tf.image_summary('output', output_shaped, 20)
      target_shaped = tf.reshape(tf.to_float(targets) * 255, [1, -1, 10, 1])
      tf.image_summary('target', target_shaped, 20)
      _activation_summary(targets, name_tensor="targets")

  # Calculate accuracy
  with tf.name_scope("accuracy"):
      predictions = tf.argmax(probs, 1, name="predictions")
      truth = tf.argmax(targets, 1, name="predictions")
      correct_predictions = tf.equal(predictions, truth)
      accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
      tf.scalar_summary("accuracy", accuracy)

  # Get the loss
  losses = tf.nn.softmax_cross_entropy_with_logits(output, tf.to_float(targets))
  mean_loss = tf.reduce_mean(losses, name="mean_loss")

  return probs, mean_loss

