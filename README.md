# kaggle-dogbreed
Participation in the kaggle dogbreed identification competition

## Getting started
The first thing we need to do is download the data. The data can be found on the kaggle project [page](https://www.kaggle.com/c/dog-breed-identification/data). Once downloaded, place the train and test directories in the top level directory for this repo. Note that the 2 other files (labels, submission sample) are already in the repo, so no download is required.

## Retraining Inception's Final Layer
The first approach we will use is an instance of transfer learning, where we use the fully trained neural network layers that are trained on a set of classes (in this case we will ise imagenet and inception) and then retrain the final layers of the network to classify new classes. This shortcut does not give us the full performance of retraining the entire model from scratch, but it is a good compromise since it is a lot faster than training from scratch. A good explanation and tutorial on transfer learning can be found [here](https://www.tensorflow.org/tutorials/image_retraining). We will use tensorflow's retraining script for training this model.
Before we can start, we need to format our data into the format that the tensorflow retraining script expects. The [convertFormat.py](code/convertFormat.py) script will do just that. Make sure that the train image folder (downloaded from kaggle) is uncompressed and in the top directory of this repo, then run the convertFormat.py script.
```BASH
python scripts/convertFormat.py
```
The script will create a directory for each dog breed and copy the images from the train directory into the appropriate dog breed directory, which is determined by the labels.csv file.

### Running the re-training script
(Note: I am not the author of the retrain.py script, and a much more detailed tutorial on retraining can be found [here](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0))
In a shell, run the following commands in the same order. first lets pick the architecture, as mentioned above we will use inception:
```BASH
ARCHITECTURE="inception_v3"
```
The next command will start a tensor board in the background, which will allow us to monitor the training progress.

```BASH
tensorboard --logdir tf_files/training_summaries &
```
Now we will rerun the training step. (To understand the different parameters passed to the retain script, use the -h option: `python -m scripts.retrain -h`). We will run the script with the following parameters:
```BASH
python -m scripts.retrain \
--bottleneck_dir=tf_files/bottlenecks \
--how_many_training_steps=5000 \
--model_dir=tf_files/models/ \
--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
--output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--architecture="${ARCHITECTURE}" \
--image_dir=train-inception-format/
```
The script taked around 30 minutes to train, and will result in an accuracy of roughly 90%, this will differ from run to tun because there is randomness in the training. Now, if we want to perform a classification using our trained model, we use the label_image.py script as follows:
```BASH
python -m  scripts.label_image --image dog.jpg
```
Here, dog.jpg is a picture of a rotweiler, which this specific model classifies correctly with around 83% certainty. The output should be something like this:
```BASH
rottweiler 0.836632
black and tan coonhound 0.0343763
greater swiss mountain dog 0.00915611
kelpie 0.00790766
staffordshire bullterrier 0.00622047
```
Now since we want to upload our results to kaggle. We need to classify all the images in the test set and write the probabilities for all 120 classes in a single csv file. We will use the batchPredictProbs.py script which produces a csv file for the 10000 images in the test set, the script classifies the images on multiple threads to speed up the process.
```BASH
python scripts.batchPredictProbs
```
This script will take around 30 minutes to run on an average CPU. this will produce a submission.csv file which can be uploaded to kaggle.
