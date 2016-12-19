"""Loading TFRecords"""
import tensorflow as tf

# Read columns TFRecord
def get_feature_columns(max_words_context, max_words_utterance):
  """Split feature tuples into raw params"""
  feature_columns = []

  feature_columns.append(tf.contrib.layers.real_valued_column(
    column_name="context", dimension=max_words_context, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="context_len", dimension=1, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance", dimension=max_words_utterance, dtype=tf.int64))
  feature_columns.append(tf.contrib.layers.real_valued_column(
      column_name="utterance_len", dimension=1, dtype=tf.int64))

  for i in range(9):
      feature_columns.append(tf.contrib.layers.real_valued_column(
          column_name="distractor_{}".format(i), dimension=max_words_utterance, dtype=tf.int64))
      feature_columns.append(tf.contrib.layers.real_valued_column(
          column_name="distractor_{}_len".format(i), dimension=1, dtype=tf.int64))

  return set(feature_columns)


def create_input_fn(mode, input_files, batch_size, num_epochs, params):
  """Prepare the input method"""
  def input_fn():
    # Get column names and sizes
    features = tf.contrib.layers.create_feature_spec_for_parsing(
        get_feature_columns(max_words_context=params.max_context_len, max_words_utterance=params.max_utterance_len))

    # Build feature map
    feature_map = tf.contrib.learn.io.read_batch_features(
        file_pattern=input_files,
        batch_size=batch_size,
        features=features,
        reader=tf.TFRecordReader,
        randomize_input=True,
        num_epochs=num_epochs,
        queue_capacity=200000 + batch_size * 10,
        name="read_batch_features_{}".format(mode))

    # Create queue variable
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
      tf.get_variable(
        "read_batch_features_eval/file_name_queue/limit_epochs/epochs",
        initializer=tf.constant(0, dtype=tf.int64))

    # Index correct utterance (0 all the time since the Ground Truth Utterance is at the beginning)
    target = tf.zeros([batch_size, 1], dtype=tf.int64)
    return feature_map, target
  return input_fn
