import tensorflow as tf
import numpy as np
from decimal import Decimal
import os


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


if __name__ == '__main__':

    input_height = 299
    input_width = 299
    input_mean = 128
    input_std = 128

    # load graph and get desired operations
    graph = load_graph("tf_files/retrained_graph.pb")
    input_operation = graph.get_operation_by_name("import/Mul");
    output_operation = graph.get_operation_by_name("import/final_result");

    # list of paths to images
    files = ['test/' + f for f in os.listdir('test') if f[-4:] == ".jpg"]

    # create a mapping between the class labels and their array index
    labels = labelMapping = [], {}
    with open("tf_files/retrained_labels.txt",'r') as labelFile:
        labels = [label.replace('\n','') for label in labelFile.readlines()]
        labelMapping = dict(zip(labels, range(len(labels))))

    probsArray = np.zeros((len(files), len(labels)))
    probsAlphabeticalIndeces = np.argsort(labels)

    colHeaders = sorted([label.strip().replace(' ', '_') for label in labels])
    colHeaderString = 'id,' + ','.join(colHeaders) + '\n'
    print(colHeaderString)
    rowHeaders = []

    with tf.Session(graph=graph) as sess:

        testImageQueue = tf.train.string_input_producer(files)
        reader = tf.WholeFileReader()
        key, value = reader.read(testImageQueue)
        image = tf.image.decode_jpeg(value, channels = 3, name='jpeg_reader')
        float_caster = tf.cast(image, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(len(files)):
            name, preprocessedImage = sess.run([key, normalized])
            results = sess.run(output_operation.outputs[0],
                                {input_operation.outputs[0]: preprocessedImage})
            results = np.squeeze(results)
            probsArray[i] = results[probsAlphabeticalIndeces]
            rowHeaders.append(str(name).split('/')[1].replace('\'','').replace('.jpg',''))

            if i % 10 == 0:
                print('classified ', i, ' images')

        coord.request_stop()
        coord.join(threads)

        filesAlphabeticalIndeces = np.argsort(np.array(rowHeaders))
        rowHeaders = sorted(rowHeaders)
        assert(len(set(rowHeaders)) == len(rowHeaders))
        probsArray = probsArray[filesAlphabeticalIndeces]

        with open('submission.csv', 'w') as submissionFile:
            np.set_printoptions(suppress=True)
            submissionFile.write(colHeaderString)
            for i in range(len(files)):
                preds = []
                for prob in probsArray[i]:
                    precision = min(19, len(str(Decimal(prob))))
                    probTxt = str(Decimal(prob))[:precision]
                    preds.append(probTxt)
                submissionFile.write(rowHeaders[i] + ',' + ','.join(preds) + '\n')
