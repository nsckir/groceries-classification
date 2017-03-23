import cStringIO as StringIO
import datetime
import inspect
import logging
import optparse
import os
import sys
import time
import urllib2
import json
import flask
import tensorflow as tf
import werkzeug
import operator
from PIL import Image
# from inception import inception_model

import exifutil

# sys.path.append('../slim/')
from datasets import dataset_utils
from datasets import scoodit_178

from nets import inception

from preprocessing import inception_preprocessing

REPO_DIRNAME = os.path.abspath(
    os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = '/home/kiril/PycharmProjects/scoodit_image_classification/models/inception_v3/scoodit_178'
DATA_SET_DIR = '/home/kiril/PycharmProjects/scoodit_image_classification/data/processed/scoodit_178'
NUM_CLASSES = 178
NUM_TOP_CLASSES = 5
# Obtain the flask app object
app = flask.Flask(__name__)
UPLOAD_FOLDER = '/tmp/demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpe', 'jpeg'])


@app.route('/', methods=['GET', 'POST'])
def classify_index():
    string_buffer = None

    if flask.request.method == 'GET':
        url = flask.request.args.get('url')
        if url:
            logging.info('Image: %s', url)
            string_buffer = urllib2.urlopen(url).read()

        file = flask.request.args.get('file')
        if file:
            logging.info('Image: %s', file)
            string_buffer = open(file, 'rb').read()

        if not string_buffer:
            return flask.render_template('index.html', has_result=False)

    elif flask.request.method == 'POST':
        string_buffer = flask.request.stream.read()

    if not string_buffer:
        resp = flask.make_response()
        resp.status_code = 400
        return resp
    names, probabilities, time_cost, accuracy = app.clf.classify_image(string_buffer)
    # json_data = json.dumps({'predictions': names, 'probabilities': probabilities.tolist()})
    predictions = sorted(dict(zip(names, probabilities.tolist())).iteritems(), key=lambda (k, v): (v, k), reverse=True)

    json_data = json.dumps(predictions, sort_keys=True)
    return flask.make_response(json_data, 200, {'ClassificationAccuracy': accuracy})
    # return flask.make_response(u",".join(names), 200, {'ClassificationAccuracy': accuracy})


@app.route('/classify_url', methods=['GET'])
def classify_url():
    imageurl = flask.request.args.get('imageurl', '')
    try:
        bytes = urllib2.urlopen(imageurl).read()
        string_buffer = StringIO.StringIO(bytes)
        image = exifutil.open_oriented_im(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    app.logger.info('Image: %s', imageurl)
    names, probs, time_cost, accuracy = app.clf.classify_image(bytes)
    return flask.render_template(
        'index.html', has_result=True, result=[True, zip(names, probs), '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                    werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        path, extension = os.path.splitext(filename)
        if extension == '.png':
            im = Image.open(filename)
            filename = "%s.jpg" % path
            im.save(filename)

        logging.info('Saving to %s.', filename)
        image = exifutil.open_oriented_im(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )

    names, probs, time_cost, accuracy = app.clf.classify_image(
        open(os.path.join(filename), "rb").read())
    return flask.render_template(
        'index.html', has_result=True, result=[True, zip(names, probs), '%.3f' % time_cost],
        imagesrc=embed_image_html(image)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/model.ckpt.data-00000-of-00001'.format(CHECKPOINT_DIR)),
        'class_labels_file': (
            '{}/labels.txt'.format(DATA_SET_DIR)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))

    def __init__(self, model_def_file, class_labels_file):
        logging.info('Loading net and associated files...')
        slim = tf.contrib.slim
        image_size = inception.inception_v3.default_image_size

        with tf.Graph().as_default(), tf.device('cpu:0'):
            self.sess = tf.Session()
            self.image_buffer = tf.placeholder(tf.string)
            image = tf.image.decode_jpeg(self.image_buffer, channels=3)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = self.eval_image(image, image_size, image_size)
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            images = tf.expand_dims(image, 0)

            # Run inference.
            with slim.arg_scope(inception.inception_v3_tf1_arg_scope()):
                logits, _ = inception.inception_v3_tf1(images, num_classes=178, is_training=False)
            predictions = tf.nn.softmax(logits)
            # logits, predictions = inception_model.inference(
            #    images, NUM_CLASSES + 1)

            # Transform output to topK result.
            self.values, self.indices = tf.nn.top_k(
                predictions, NUM_TOP_CLASSES)

            self.init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(CHECKPOINT_DIR, 'model.ckpt'),
                slim.get_model_variables('InceptionV3'))

            # variable_averages = tf.train.ExponentialMovingAverage(
            #     inception_model.MOVING_AVERAGE_DECAY)
            # variables_to_restore = variable_averages.variables_to_restore()
            # tf.global_variables_initializer().run(session=self.sess)
            # tf.local_variables_initializer().run(session=self.sess)
            # saver = tf.train.Saver(variables_to_restore)
            # saver.restore(self.sess, model_def_file)
            # Required to get the filename matching to run.

            self.label_names = ['none']
            # with open(class_labels_file) as f:
            #     for line in f.read().decode("utf-8").splitlines():
            #         self.label_names.append(line)
            labels = scoodit_178.dataset_utils.read_label_file(DATA_SET_DIR)
            self.label_names = {k: v.rsplit('_', 1)[0].replace('_', ' ') for k, v in labels.items()}

    def eval_image(self, image, height, width, scope=None):
        """Prepare one image for evaluation.

        Args:
          image: 3-D float Tensor
          height: integer
          width: integer
          scope: Optional scope for op_scope.
        Returns:
          3-D float Tensor of prepared image.
        """
        # image = tf.reshape(image, [height,width,3])
        # return images
        with tf.op_scope([image, height, width], scope, 'eval_image'):
            # Crop the central region of the image with an area containing 87.5% of
            # the original image.
            image = tf.image.central_crop(image, central_fraction=0.875)

            # Resize the image to the original height and width.
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],
                                             align_corners=False)
            image = tf.squeeze(image, [0])
            return image

    def classify_image(self, image):
        try:
            start_time = time.time()
            self.init_fn(self.sess)
            probabilities, labels = self.sess.run(
                [self.values, self.indices], feed_dict={self.image_buffer: image})

            # with tf.Session() as sess:
            #    self.init_fn(sess)
            #    np_image, probabilities = self.sess.run([image, probabilities])
            #    probabilities = probabilities[0, 0:] * 100
            #    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            labels = labels[0]
            probabilities = probabilities[0]
            end_time = time.time()
            app.logger.info(
                "classify_image cost %.2f secs", end_time - start_time)
            return [self.label_names[labels[0]],
                    self.label_names[labels[1]],
                    self.label_names[labels[2]],
                    self.label_names[labels[3]],
                    self.label_names[labels[4]]], probabilities[:5], end_time - start_time, sum(probabilities)
        except Exception as err:
            logging.info('Classification error: %s', err)
            return None


def setup_app(app):
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.logger.info('testing sample picture...')
    bytes = open(os.path.join(REPO_DIRNAME, "sample.JPEG"), "rb").read()
    ret, _, _, _ = app.clf.classify_image(bytes)
    app.logger.info("sample testing complete %s %s %s", ret[0], ret[1], ret[2])


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5005)
    opts, args = parser.parse_args()
    # Initialize classifier + warm start by forward for allocation
    setup_app(app)
    app.run(debug=True, processes=1, host='0.0.0.0', port=opts.port)


logging.getLogger().setLevel(logging.INFO)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if __name__ == '__main__':
    start_from_terminal(app)
else:
    gunicorn_error_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_error_logger.handlers
    app.logger.setLevel(logging.INFO)
    setup_app(app)
