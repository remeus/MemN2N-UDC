"""LSTM model"""
import tensorflow as tf
from models import helpers


FLAGS = tf.flags.FLAGS

def get_embeddings(hparams, glove):
  if glove:
    tf.logging.info("Loading GloVe embedding ...")
    glove_path = "data/glove.6B.100d.txt"
    vocab_path = "data/vocabulary.txt"
    vocab_array, vocab_dict = helpers.load_vocab(vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors, hparams.embedding_dim)
  else:
    tf.logging.info("Loading random embedding ...")
    initializer = tf.random_uniform_initializer(-0.25, 0.25)

  return tf.get_variable(
    "word_embeddings",
    shape=[hparams.vocab_size, hparams.embedding_dim],
    initializer=initializer)


def lstm(hparams, mode, context, context_len, utterance, utterance_len, targets):

  # Only n_utt first examples are considered to speed up training
  if mode == tf.contrib.learn.ModeKeys.TRAIN:
      n_utt = 10 # Between 1 and 10
  else:
      n_utt = 10 # Keep it to 10 for eval

  # Restructure data for the network
  context = tf.tile(context, [n_utt, 1])
  context_len = tf.tile(context_len, [n_utt])
  utterances = tf.unpack(utterance, num=10, axis=1)
  utterance = tf.concat(0, utterances[0:n_utt])
  utterance_len = tf.unpack(utterance_len, num=10, axis=1)
  utterance_len = tf.concat(0, utterance_len[0:n_utt])
  targetss = tf.unpack(targets, num=10, axis=1)
  targets = tf.concat(0, targetss[0:n_utt])
  targets = tf.expand_dims(targets, 1)

  # Initialize embeddings randomly or with pre-trained vectors if available
  embeddings_W = get_embeddings(hparams, glove=False)

  # Embed the context and the utterance
  context_embedded = tf.nn.embedding_lookup(
      embeddings_W, context, name="embed_context")
  utterance_embedded = tf.nn.embedding_lookup(
      embeddings_W, utterance, name="embed_utterance")

  # Build the RNN
  rnn_dim = 256
  with tf.variable_scope("rnn"):
    # We use an LSTM Cell
    cell = tf.nn.rnn_cell.LSTMCell(
        rnn_dim,
        forget_bias=2.0,
        use_peepholes=True,
        state_is_tuple=True)

    # Run the utterance and context through the RNN
    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
        cell,
        tf.concat(0, [context_embedded, utterance_embedded]),
        sequence_length=tf.concat(0, [context_len, utterance_len]),
        dtype=tf.float32)
    encoding_context, encoding_utterance = tf.split(0, 2, rnn_states.h)

  with tf.variable_scope("prediction"):
    M = tf.get_variable("M",
      shape=[rnn_dim, rnn_dim],
      initializer=tf.truncated_normal_initializer())

    # "Predict" a  response: c * M
    generated_response = tf.matmul(encoding_context, M)
    generated_response = tf.expand_dims(generated_response, 2)
    encoding_utterance = tf.expand_dims(encoding_utterance, 2)

    # Dot product between generated response and actual response
    # (c * M) * r
    logits = tf.batch_matmul(generated_response, encoding_utterance, True)
    logits = tf.squeeze(logits, [2])

    # Apply sigmoid to convert logits to probabilities
    probs = tf.sigmoid(logits)

    if mode == tf.contrib.learn.ModeKeys.INFER:
      return probs, None

    # Display monitoring
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        probs_shaped = tf.reshape(probs * 255, [1, -1, n_utt, 1])
        tf.image_summary('probs', probs_shaped, 20)
        output_shaped = tf.reshape(logits, [1, -1, n_utt, 1])
        tf.image_summary('output', output_shaped, 20)
        target_shaped = tf.reshape(tf.to_float(targets) * 255, [1, -1, n_utt, 1])
        tf.image_summary('target', target_shaped, 20)

    # Calculate the binary cross-entropy loss
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(targets))

    if mode == tf.contrib.learn.ModeKeys.EVAL:
        split_probs = tf.split(0, 10, probs)
        probs = tf.concat(1, split_probs)

  # Mean loss across the batch of examples
  mean_loss = tf.reduce_mean(losses, name="mean_loss")

  return probs, mean_loss
