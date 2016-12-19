"""Prepare and update the model"""
import tensorflow as tf


def get_id_feature(features, key, len_key, max_len):
  """Return context/utterance sentences and corresponding number of words"""
  ids = features[key]
  ids_len = tf.squeeze(features[len_key], [1])
  ids_len = tf.minimum(ids_len, tf.constant(max_len, dtype=tf.int64))
  return ids, ids_len


def create_train_op(loss, hparams):
  """Apply Backpropagation"""
  def _learning_rate_decay_fn(learning_rate, global_step):
      return tf.train.exponential_decay(
          learning_rate,
          global_step,
          decay_steps=1000,
          decay_rate=0.5,
          staircase=True)
  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=hparams.learning_rate,
      clip_gradients=hparams.max_grad_norm,
      optimizer=hparams.optimizer,
      # gradient_noise_scale=1e-3, # Optional
      # learning_rate_decay_fn=_learning_rate_decay_fn # Optional
  )
  return train_op


def create_model_fn(hparams, model_impl):
  """Prepare the model input"""

  def model_fn(features, targets, mode):

    # Fetch features
    context, context_len = get_id_feature(
        features, "context", "context_len", hparams.max_context_len)

    utterance, utterance_len = get_id_feature(
        features, "utterance", "utterance_len", hparams.max_utterance_len)

    batch_size = targets.get_shape().as_list()[0]

    all_utterances = [utterance]
    all_utterance_lens = [utterance_len]
    if mode != tf.contrib.learn.ModeKeys.INFER:
        all_targets = [tf.ones([batch_size], dtype=tf.int64)]

    for i in range(9):
        distractor, distractor_len = get_id_feature(features,
                                                    "distractor_{}".format(i),
                                                    "distractor_{}_len".format(i),
                                                    hparams.max_utterance_len)
        all_utterances.append(distractor)
        all_utterance_lens.append(distractor_len)
        if mode != tf.contrib.learn.ModeKeys.INFER:
            all_targets.append(
                tf.zeros([batch_size], dtype=tf.int64)
            )

    # Predict
    if mode == tf.contrib.learn.ModeKeys.INFER:
        all_utterances = tf.pack(all_utterances, axis=1)
        all_utterance_lens = tf.pack(all_utterance_lens, axis=1)
        probs, loss = model_impl(
            hparams,
            mode,
            context,
            context_len,
            all_utterances,
            all_utterance_lens,
            None)
        return probs, 0.0, None

    # PRandomize utterance order
    elif mode == tf.contrib.learn.ModeKeys.TRAIN:
        all_utterances = tf.pack(all_utterances, axis=0)
        all_utterance_lens = tf.pack(all_utterance_lens, axis=0)
        all_targets = tf.pack(all_targets, axis=0)

        all_utterance_lens = tf.expand_dims(all_utterance_lens, 2)
        all_targets = tf.expand_dims(all_targets, 2)
        all = tf.concat(2, [all_utterances, all_utterance_lens, all_targets])
        alls = []
        s = hparams.max_utterance_len
        for i in range(batch_size):
            all_i = tf.slice(all, [0, i, 0], [10, 1, s+2])
            all_i = tf.random_shuffle(all_i)
            alls.append(all_i)
        all = tf.pack(alls, axis=1)
        all = tf.squeeze(all, [2])
        all_utterances = tf.slice(all, [0, 0, 0], [10, -1, s])
        all_utterance_lens = tf.slice(all, [0, 0, s], [10, -1, 1])
        all_targets = tf.slice(all, [0, 0, s+1], [10, -1, 1])
        all_utterances = tf.transpose(all_utterances, [1, 0, 2])
        all_utterance_lens = tf.transpose(all_utterance_lens, [1, 0, 2])
        all_targets = tf.transpose(all_targets, [1, 0, 2])
        all_utterance_lens = tf.squeeze(all_utterance_lens, [2])
        all_targets = tf.squeeze(all_targets, [2])

        # Compute the output and loss through the model
        probs, loss = model_impl(
            hparams,
            mode,
            context,
            context_len,
            all_utterances,
            all_utterance_lens,
            all_targets)

        train_op = create_train_op(loss, hparams)
        return probs, loss, train_op

    elif mode == tf.contrib.learn.ModeKeys.EVAL:

        all_utterances = tf.pack(all_utterances, axis=1)
        all_utterance_lens = tf.pack(all_utterance_lens, axis=1)
        all_targets = tf.pack(all_targets, axis=1)

        probs, loss = model_impl(
            hparams,
            mode,
            context,
            context_len,
            all_utterances,
            all_utterance_lens,
            all_targets)

        return probs, loss, None


  return model_fn
