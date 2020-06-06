import os
import sys

filename = sys.argv[1]
metric_name = sys.argv[2] + ':'

metrics = []
with open(filename) as f:
    for line in f:
        if 'Iter' not in line:
            continue
        fields = line.split(',')
        for field in fields:
            #if 'Iter' in field:
            #    print(field.split(' ')[-1], end=' ')
            print(fields)
            if metric_name in field:
                print(field.split(' ')[-1])

