from django.db import models
import json
import os

class Result(models.Model):
    id = models.AutoField(primary_key=True)
    reaction = models.IntegerField(null=False)
    relation = models.IntegerField(null=False)
    action = models.IntegerField(null=False)
    person_1 = models.IntegerField(null=False)
    person_2 = models.IntegerField(null=False)
    moral = models.BooleanField(null=False)

    def __str__(self):
        options = json.load(open(os.path.join(os.path.dirname(__file__), 'sentence-options.json')))

        try:
            victim = options['relation'][self.relation] % (options['name'][self.person_1])
        except TypeError:
            victim = options['relation'][self.relation]
        now = options['reaction'][self.reaction] % (options['name'][self.person_2])
        past = options['action'][self.action] % (victim)
        sentence = '%s: %s %s, weil %s %s' % (
            'Gut' if self.moral else 'Bad',
            options['name'][self.person_1],
            now,
            options['name'][self.person_2],
            past,
        )
        return sentence

