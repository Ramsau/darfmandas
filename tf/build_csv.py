import json
import os
import csv
from random import shuffle

path_data = os.path.join(os.path.dirname(__file__), 'dump.json')
path_options = os.path.join(os.path.dirname(__file__), 'sentence-options.json')
test_csv = os.path.join(os.path.dirname(__file__), 'test.csv')
train_csv = os.path.join(os.path.dirname(__file__), 'train.csv')
data = [x['fields'] for x in json.load(open(path_data)) if x['model'] == 'poll.result']
shuffle(data)
options = json.load(open(path_options))

option_count = len(options['reaction']) + len(options['relation']) + len(options['action'])
reaction_count = len(options['reaction'])
relation_count = len(options['relation'])
action_count = len(options['action'])


def writeLine(entry):
    try:
        reactions = [0] * reaction_count
        reactions[entry['reaction']] = 1
        relations = [0] * relation_count
        relations[entry['relation']] = 1
        actions = [0] * action_count
        actions[entry['action']] = 1
        line = reactions + relations + actions + [1 if entry['moral'] else 0]
        writer.writerow(line)
    except IndexError:
        print('Bad entry: %s' % (str(entry)))

with open(test_csv, 'w+') as test:
    writer = csv.writer(test, delimiter=',')
    first_line = [
        int(len(data)*0.2),
        option_count + 1,
        'immoral',
        'moral'
    ]
    writer.writerow(first_line)
    for entry in data[0:int(len(data)*0.2)]:
        writeLine(entry)


with open(train_csv, 'w+') as train:
    writer = csv.writer(train, delimiter=',')
    first_line = [
        int(len(data)),
        option_count + 1,
        'immoral',
        'moral'
    ]
    writer.writerow(first_line)
    for entry in data:
        writeLine(entry)

