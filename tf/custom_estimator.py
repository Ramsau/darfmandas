from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import get_data



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=500, type=int, help='batch size')
parser.add_argument('--train_steps', default=1, type=int,
                    help='number of training steps')
batch_size = 500
train_steps = 1
feature_spec = {
    'Reaction1': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction2': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction3': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction4': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction5': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction6': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction7': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction8': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction9': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction10': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction11': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction12': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction13': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction14': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction15': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction16': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Reaction17': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation1': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation2': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation3': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation4': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation5': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation6': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation7': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation8': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation9': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation10': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation11': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation12': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation13': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation14': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation15': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Relation16': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action1': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action2': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action3': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action4': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action5': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action6': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action7': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action8': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action9': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action10': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action11': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action12': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action13': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action14': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action15': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action16': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action17': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action18': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
    'Action19': tf.FixedLenFeature(shape=(1,), default_value=0, dtype=tf.float32, ),
}

def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def serving_input_receiver_fn():
    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                           shape=[batch_size],
                                           name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()

    (train_x, train_y), (test_x, test_y) = get_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    print(my_feature_columns)
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [100, 100, 100, 100],
            'n_classes': 2,
        },
        model_dir='/models/angela'
    )

    classifier.train(
        input_fn=lambda:get_data.train_input_fn(train_x, train_y, batch_size),
        steps=train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda:get_data.eval_input_fn(test_x, test_y, batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    expected = ['moral', 'immoral', 'immoral']
    predict_x = {
        'Reaction1': [0, 0, 0],
        'Reaction2': [0, 0, 0],
        'Reaction3': [0, 0, 0],
        'Reaction4': [0, 0, 0],
        'Reaction5': [0, 0, 0],
        'Reaction6': [0, 0, 0],
        'Reaction7': [0, 1, 1],
        'Reaction8': [0, 0, 0],
        'Reaction9': [1, 0, 0],
        'Reaction10': [0, 0, 0],
        'Reaction11': [0, 0, 0],
        'Reaction12': [0, 0, 0],
        'Reaction13': [0, 0, 0],
        'Reaction14': [0, 0, 0],
        'Reaction15': [0, 0, 0],
        'Reaction16': [0, 0, 0],
        'Reaction17': [0, 0, 0],
        'Relation1': [0, 0, 0],
        'Relation2': [0, 0, 0],
        'Relation3': [0, 0, 0],
        'Relation4': [0, 0, 0],
        'Relation5': [0, 0, 0],
        'Relation6': [1, 0, 0],
        'Relation7': [0, 0, 0],
        'Relation8': [0, 0, 0],
        'Relation9': [0, 0, 0],
        'Relation10': [0, 1, 0],
        'Relation11': [0, 0, 0],
        'Relation12': [0, 0, 0],
        'Relation13': [0, 0, 1],
        'Relation14': [0, 0, 0],
        'Relation15': [0, 0, 0],
        'Relation16': [0, 0, 0],
        'Action1': [0, 0, 1],
        'Action2': [0, 0, 0],
        'Action3': [0, 0, 0],
        'Action4': [0, 0, 0],
        'Action5': [0, 0, 0],
        'Action6': [1, 0, 0],
        'Action7': [0, 0, 0],
        'Action8': [0, 0, 0],
        'Action9': [0, 0, 0],
        'Action10': [0, 0, 0],
        'Action11': [0, 1, 0],
        'Action12': [0, 0, 0],
        'Action13': [0, 0, 0],
        'Action14': [0, 0, 0],
        'Action15': [0, 0, 0],
        'Action16': [0, 0, 0],
        'Action17': [0, 0, 0],
        'Action18': [0, 0, 0],
        'Action19': [0, 0, 0],
    }

    predictions = classifier.predict(
        input_fn=lambda:get_data.eval_input_fn(predict_x,
                                                labels=None,
                                                batch_size=batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(get_data.MORALITY[class_id],
                              100 * probability, expec))

    classifier.export_savedmodel('/models/angela-saved/', serving_input_receiver_fn)
