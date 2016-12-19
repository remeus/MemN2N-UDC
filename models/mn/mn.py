"""The Memory Network model"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from models import helpers


def _activation_summary(x, name_tensor=None):
    """Helper to create summaries for activations."""
    if name_tensor is None:
        name_tensor = x.op.name
    tf.histogram_summary(name_tensor + '/activations', x)
    tf.scalar_summary(name_tensor + '/sparsity', tf.nn.zero_fraction(x))


def position_encoding(sentence_size, embedding_size):
    """Position Encoding described in End-To-End Memory Network paper"""
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def normal_encoding(sentence_size, embedding_size):
    """Unit encoding"""
    encoding = np.ones((sentence_size, embedding_size), dtype=np.float32)
    return encoding


def glove_init(embedding_size):
    """GloVe initialization"""
    glove_path = "data/glove.6B.100d.txt"
    vocab_path = "data/vocabulary.txt"
    tf.logging.info("Loading GloVe embeddings ...")
    vocab_array, vocab_dict = helpers.load_vocab(vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             embedding_size)
    return initializer



class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, vocab_size, batch_size, sentence_size, memory_size, embedding_size,
                 hops, nonlin, initializer, encode_position, glove, name='MemN2N'):
        """Creates an End-To-End Memory Network

        Args:

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            batch_size: The batch size.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            encode_position: A boolean corresponding to the encoding method chosen.

            glove: A boolean corresponding to the initialization method chosen.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._vocab_size = vocab_size
        self._batch_size = batch_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._nonlin = nonlin
        self._init = initializer
        self._name = name

        # Build variables
        self._build_vars(glove)

        if encode_position:
            self._encoding = tf.constant(position_encoding(self._sentence_size, self._embedding_size), name="encoding")
        else:
            self._encoding = tf.constant(normal_encoding(self._sentence_size, self._embedding_size), name="encoding")


    def _build_vars(self, glove):
        """Build variables"""
        with tf.variable_scope(self._name):
            if glove:
                glove_in = glove_init(self._embedding_size)
                self.A = tf.get_variable("A", shape=[self._vocab_size, self._embedding_size], initializer=glove_in)
                self.B = tf.get_variable("B", shape=[self._vocab_size, self._embedding_size], initializer=glove_in)
            else:
                nil_word_slot = tf.zeros([3, self._embedding_size])
                A = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-3, self._embedding_size]) ]) # TODO
                B = tf.concat(0, [ nil_word_slot, self._init([self._vocab_size-3, self._embedding_size]) ])
                self.A = tf.Variable(A, name="A")
                self.B = tf.Variable(B, name="B")

            _activation_summary(self.A)
            _activation_summary(self.B)

            # Temporal encoding
            # self.TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA') # optional


    def inference(self, stories, queries):
        """Compute output"""
        with tf.variable_scope(self._name):

            # We embed the context and neglect the order of words
            q_emb = tf.nn.embedding_lookup(self.B, queries)
            _activation_summary(q_emb, "q_emb")
            u = tf.reduce_sum(q_emb * self._encoding, 1)
            _activation_summary(u, "u_0")
            u = tf.transpose(tf.expand_dims(u, -1), [0, 2, 1])

            # We embed the utterances and neglect the order of words
            m_emb = tf.nn.embedding_lookup(self.A, stories)
            m = tf.reduce_sum(m_emb * self._encoding, 2)  # + self.TA # TODO
            _activation_summary(m, "m")

            probs = tf.zeros([self._batch_size, self._memory_size])

            # Loop over memory
            for _ in range(self._hops):

                # We compute the match between the internal context and utterance representations
                probs += tf.reduce_sum(m * u, 2)

                # We update the utterance internal state
                m = tf.expand_dims(tf.nn.softmax(probs), -1) * m

            return probs / self._hops
