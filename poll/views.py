from django.shortcuts import render
import json
import os
import random
from django.http import JsonResponse
from poll.models import *
import tf.get_data
from tf.estimate import predict


def poll(request):
    if not request.GET:
        sentence, name, reaction, relation, action, person1, person2, prediction = createSentence()
        return render(request, "link.html", {'sentence': sentence, 'name': name, 'reaction': reaction, 'relation': relation,
                                            'action': action, 'person1': person1, 'person2': person2,
                                             'htmlfile': "main.html", 'prediction': moralAssessment(prediction)})
    else:
        print(request.GET)
        moral = True if request.GET['moral'] == 'true' else False
        sentence = request.GET['sentence']
        newEntry = Result.objects.create(reaction=request.GET['reaction'], relation=request.GET['relation'],
                                         action=request.GET['action'], person_1=request.GET['person1'],
                                         person_2=request.GET['person2'], moral=moral)
        newEntry.save()

        return render(request, 'link.html', {'sentence': sentence, 'moral': moral,
                            'prediction': request.GET['prediction'], 'htmlfile': "after_answer.html"})

def about(request):
    return render(request, "about.html")

def createSentence():
    options = json.load(open(os.path.join(os.path.dirname(__file__), 'sentence-options.json')))

    reaction = random.choice(options['reaction'])
    relation = random.choice(options['relation'])
    action = random.choice(options['action'])
    person2 = person1 = random.choice(options['name'])
    while person2 == person1:
        person2 = random.choice(options['name'])

    try:
        victim = relation % (person1)
    except TypeError:
        victim = relation
    now = reaction % (person2)
    past = action % (victim)
    sentence = '%s %s, weil %s %s.' %(
        person1,
        now,
        person2,
        past,
    )
    reaction_num = options['reaction'].index(reaction)
    relation_num = options['relation'].index(relation)
    action_num = options['action'].index(action)

    prediction = predict(reaction_num, relation_num, action_num)

    return sentence, person1, reaction_num, relation_num, action_num,\
        options['name'].index(person1), options['name'].index(person2), prediction

def moralAssessment(prediction):
    val = prediction[0] if prediction[0] > prediction[1] else prediction[1]
    if 0.5 <= val < 0.7:
        flavor = random.choice(['Ich glaube, das ist', 'Ich bin mir nicht sicher, vielleicht ist es', 'Das ist eher'])
    elif 0.7 <= val < 0.9:
        flavor = random.choice(['Wahrscheinlich ist das', 'Ich bin mir ziemlich sicher, das ist', 'Das ist doch wohl'])
    else:
        flavor = random.choice(['Das ist doch offensichtlich', 'Ist doch klar, das ist', 'Ich bin mir sicher, das ist', 'Das ist sicherlich'])
    if prediction[0] > prediction[1]:
        string = '{0} nicht moralisch ({1:0.1f}%)'.format(flavor, prediction[0]*100)
    elif prediction[0] < prediction[1]:
        string = '{0} moralisch ({1:0.1f}%)'.format(flavor, prediction[1]*100)
    else:
        string = 'Ich kann mich nicht entscheiden(50%/50%)'
    return string

