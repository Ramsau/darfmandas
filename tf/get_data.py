import pandas as pd
import tensorflow as tf
import os
import json

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'train.csv')
TEST_PATH = os.path.join(os.path.dirname(__file__), 'test.csv')
options_path = os.path.join(os.path.dirname(__file__), 'sentence-options.json')
options = json.load(open(options_path))

CSV_COLUMN_NAMES = ['Reaction' + str(x+1) for x, i in enumerate(options['reaction'])] +\
                    ['Relation' + str(x+1) for x, i in enumerate(options['relation'])] +\
                    ['Action' + str(x+1) for x, i in enumerate(options['action'])] +\
                    ['Morality']

print(CSV_COLUMN_NAMES)
MORALITY = ['Immoral', 'Moral']

def load_data(y_name='Morality'):
    """Returns the iris dataset as (train_x, train_y), (test_x, test_y)."""

    train = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0]] * 53

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Morality')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset
