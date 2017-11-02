import os
from shutil import copyfile

labels = open('labels.csv').readlines()[1:]
imageNames = [line.split(',')[0] for line in labels]
classes = [line.split(',')[1].replace('\n','') for line in labels]
labels = list(zip(imageNames, classes))
[os.makedirs('train-inception-format/' + classLabel + '/', exist_ok=True) \
                                    for classLabel in set(classes)]

for image, classLabel in labels:
    src = 'train/' + image + '.jpg'
    dst = 'train-inception-format/' + classLabel + '/' + image + '.jpg'
    copyfile(src, dst)
