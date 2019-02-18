from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tfl
import tf.get_data

feature_spec = {
    'Reaction1': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction2': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction3': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction4': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction5': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction6': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction7': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction8': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction9': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction10': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction11': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction12': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction13': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction14': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction15': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction16': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Reaction17': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation1': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation2': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation3': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation4': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation5': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation6': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation7': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation8': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation9': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation10': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation11': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation12': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation13': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation14': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation15': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Relation16': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action1': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action2': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action3': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action4': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action5': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action6': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action7': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action8': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action9': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action10': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action11': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action12': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action13': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action14': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action15': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action16': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action17': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action18': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
    'Action19': tfl.FixedLenFeature(shape=(1,), default_value=0, dtype=tfl.float32, ),
}

def my_model(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.1."""
    # Create three fully connected layers.
    net = tfl.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tfl.layers.dense(net, units=units, activation=tfl.nn.relu)

    # Compute logits (1 per class).
    logits = tfl.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tfl.argmax(logits, 1)
    if mode == tfl.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tfl.newaxis],
            'probabilities': tfl.nn.softmax(logits),
            'logits': logits,
        }
        return tfl.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tfl.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tfl.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tfl.summary.scalar('accuracy', accuracy[1])

    if mode == tfl.estimator.ModeKeys.EVAL:
        return tfl.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tfl.estimator.ModeKeys.TRAIN

    optimizer = tfl.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tfl.train.get_global_step())
    return tfl.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


graph = tfl.Graph()
with graph.as_default():
    sess = tfl.Session()

    # Generate predictions from the model
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

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in feature_spec.keys():
        my_feature_columns.append(tfl.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tfl.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [100, 100, 100, 100],
            # The model must choose between 2 classes.
            'n_classes': 2,
        },
        model_dir='/models/angela'
    )


def predict_base(input_x):
    predictions = classifier.predict(
        input_fn=lambda:tf.get_data.eval_input_fn(input_x,
                                                labels=None,
                                                batch_size=1))

    return predictions

def predict(reaction, relation, action):
    rects = [[0] for x in range(17)]
    rects[reaction] = [1]
    rels = [[0] for x in range(16)]
    rels[relation] = [1]
    acts = [[0] for x in range(19)]
    acts[action] = [1]
    vals = rects + rels + acts
    input_x = dict(zip(predict_x.keys(), vals))
    predictions = predict_base(input_x)
    for pred in predictions:
        prediction = pred
    return prediction['probabilities']


probs = predict(0,0,9)
print(probs)
