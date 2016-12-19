# Main code written by Shuaiyi Liu (https://github.com/ShuaiyiLiu/)
"""Implement the DSSM for context-utterance matching"""
import tensorflow as tf
from models import helpers


def glove_init(embedding_size):
    """GloVe intialization"""
    glove_path = "data/glove.6B.100d.txt"
    vocab_path = "data/vocabulary.txt"
    tf.logging.info("Loading GloVe embeddings ...")
    vocab_array, vocab_dict = helpers.load_vocab(vocab_path)
    glove_vectors, glove_dict = helpers.load_glove_vectors(glove_path, vocab=set(vocab_array))
    initializer = helpers.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             embedding_size)
    return initializer


def sent_cnn(input_x_u,
            input_x_r,
            sequence_length,
            num_classes,
            embedding_size,
            voc_size,
            filter_sizes,
            num_filters,
            batch_size,
            embeddings_trainable,
            glove,
            dropout_keep_prob=0.5):
        """Compute the output of the DSSM"""

        # Word embedding initialization
        if glove:
            init = glove_init(embedding_size)
        else:
            init = tf.random_normal_initializer(stddev=0.1)

        # Embedding layer
        with tf.name_scope("embedding"):
            W = tf.get_variable("W", shape=[voc_size, embedding_size], initializer=init,
                                trainable=embeddings_trainable, dtype=tf.float32)

            # batch_size x sequence_length x embedding_size
            embedded_u = tf.nn.embedding_lookup(W, input_x_u)
            print("DEBUG: embedded_u -> %s" % embedded_u)
            # batch_size x num_classes x sequence_length x embedding_size
            embedded_r = tf.nn.embedding_lookup(W, input_x_r)
            print("DEBUG: embedded_r -> %s" % embedded_r)
            # batch_size x sequence_length x embedding_size x 1
            embedded_u_expanded = tf.expand_dims(embedded_u, -1)
            print("DEBUG: embedded_u_expanded -> %s" % embedded_u_expanded)
            # batch_size x num_classes x sequence_length x embedding_size x 1
            embedded_r_expanded = tf.expand_dims(embedded_r, -1)
            print("DEBUG: embedded_r_expanded -> %s" % embedded_r_expanded)

        # Create a convolution + maxpooling layer for each filter size
        pooled_outputs_u = []
        pooled_outputs_r = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s-u" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),
                                name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name='b')
                conv_u = tf.nn.conv2d(
                    embedded_u_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-u")
                # Apply nonlinearity
                h_u = tf.nn.sigmoid(tf.nn.bias_add(conv_u, b), name="activation-u")

                # Maxpooling over outputs
                pooled_u = tf.nn.max_pool(
                    h_u,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool-u")
                pooled_outputs_u.append(pooled_u)

                # Pass each element in x_r through the same layer
                pooled_outputs_r_wclasses = []
                for j in range(num_classes):
                    embedded_r = embedded_r_expanded[:, j, :, :, :]
                    conv_r_j = tf.nn.conv2d(
                        embedded_r,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv-r-%s" % j)

                    h_r_j = tf.nn.sigmoid(tf.nn.bias_add(conv_r_j, b), name="activation-r-%s" % j)

                    pooled_r_j = tf.nn.max_pool(
                        h_r_j,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="pool-r-%s" % j)
                    pooled_outputs_r_wclasses.append(pooled_r_j)
                # out_tensor: batch_size x 1 x num_class x num_filters
                out_tensor = tf.concat(2, pooled_outputs_r_wclasses)
                pooled_outputs_r.append(out_tensor)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print("DEBUG: pooled_outputs_u -> %s" % pooled_outputs_u)
        h_pool_u = tf.concat(3, pooled_outputs_u)
        print("DEBUG: h_pool_u -> %s" % h_pool_u)
        # batch_size x 1 x num_filters_total
        h_pool_flat_u = tf.reshape(h_pool_u, [-1, 1, num_filters_total])
        print("DEBUG: h_pool_flat_u -> %s" % h_pool_flat_u)

        print("DEBUG: pooled_outputs_r -> %s" % pooled_outputs_r)
        h_pool_r = tf.concat(3, pooled_outputs_r)
        print("DEBUG: h_pool_r -> %s" % h_pool_r)
        # h_pool_flat_r: batch_size x num_classes X num_filters_total
        h_pool_flat_r = tf.reshape(h_pool_r, [-1, num_classes, num_filters_total])
        print("DEBUG: h_pool_flat_r -> %s" % h_pool_flat_r)

        # Add dropout layer to avoid overfitting
        with tf.name_scope("dropout"):
            h_features = tf.concat(1, [h_pool_flat_u, h_pool_flat_r])
            print("DEBUG: h_features -> %s" % h_features)
            h_features_dropped = tf.nn.dropout(h_features,
                                               dropout_keep_prob,
                                               noise_shape=[batch_size, 1, num_filters_total])

            h_dropped_u = h_features_dropped[:, :1, :]
            h_dropped_r = h_features_dropped[:, 1:, :]

        # cosine layer - final scores and predictions
        with tf.name_scope("cosine_layer"):
            dot = tf.reduce_sum(tf.mul(h_dropped_u, h_dropped_r), 2)
            print("DEBUG: dot -> %s" % dot)
            sqrt_u = tf.sqrt(tf.reduce_sum(h_dropped_u ** 2, 2))
            print("DEBUG: sqrt_u -> %s" % sqrt_u)
            sqrt_r = tf.sqrt(tf.reduce_sum(h_dropped_r ** 2, 2))
            print("DEBUG: sqrt_r -> %s" % sqrt_r)
            epsilon = 1e-5
            cosine = tf.maximum(dot / (tf.maximum(sqrt_u * sqrt_r, epsilon)), epsilon)
            print("DEBUG: cosine -> %s" % cosine)
            output = 100 * cosine

        return output


