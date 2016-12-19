"""Manage the DSSM model"""
import tensorflow as tf
from models.dssm.dssm import sent_cnn

FLAGS = tf.flags.FLAGS


def _activation_summary(x, name_tensor=None):
    """Helper to create summaries for activations."""
    if name_tensor is None:
        name_tensor = x.op.name
    tf.histogram_summary(name_tensor + '/activations', x)
    tf.scalar_summary(name_tensor + '/sparsity', tf.nn.zero_fraction(x))


def dssm(hparams, mode, context, context_len, utterance, utterance_len, targets):

  # Build the DSSM model
  batch_size = targets.get_shape().as_list()[0]
  output = sent_cnn(context,
                    utterance,
                    sequence_length=hparams.max_context_len,
                    num_classes=10,
                    embedding_size=hparams.embedding_dim,
                    voc_size=hparams.vocab_size,
                    filter_sizes=[2, 3, 4],
                    num_filters=10,
                    batch_size=batch_size,
                    embeddings_trainable=True,
                    glove=True,
                    dropout_keep_prob=0.85)
  _activation_summary(output, "output")

  # Get the probabilities
  probs = tf.nn.softmax(output, name="probs")
  _activation_summary(probs)

  if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None
  else:
      # Display the monitoring
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
          truth =  tf.argmax(targets, 1, name="predictions")
          correct_predictions = tf.equal(predictions, truth)
          accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
          tf.scalar_summary("accuracy", accuracy)

  # Get the loss
  losses = tf.nn.softmax_cross_entropy_with_logits(output, tf.to_float(targets))
  mean_loss = tf.reduce_mean(losses, name="mean_loss")

  return probs, mean_loss

































