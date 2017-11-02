# kaggle-dogbreed
Participation in the kaggle dogbreed identification competition

## Getting started
The first thing we need to do is download the data. The data can be found on the kaggle project [page](https://www.kaggle.com/c/dog-breed-identification/data). Once downloaded, place the train and test directories in the top level directory for this repo. Note that the 2 other files (labels, submission sample) are already in the repo, so no download is required.

## Retraining Inception's Final Layer
The first approach we will use is an instance of transfer learning, where we use the fully trained neural network layers that is trained on a set of classes (in this case we will ise imagenet and inception) and then retrain the final layers of the network to classify new classes. This shortcut does not give us the full performance of retraining the entire model from scratch, but it is a good compromise since it is a lot faster than training from scratch. A good explanation and tutorial on transfer learning can be found [here](https://www.tensorflow.org/tutorials/image_retraining). We will use tensorflow's retraining script for training this model.
Before we can start, we need to format our data into the format that the tensorflow retraining script expects. The [convertFormat.py](code/convertFormat.py) script will do just that. Make sure that the train image folder is uncompressed and in the top directory of this repo, then run the convertFormat.py script.
```BASH
python scripts/convertFormat.py
```
The script will create a directory for each dog breed and copy the images from the train directory into the appropriate dog breed directory, which is determined by the labels.csv file.
