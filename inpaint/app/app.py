#!/usr/bin/env python3
import argparse
import os
import random
import sys
from io import BytesIO

import numpy as np
import prometheus_client as prometheus
import prometheus_client.multiprocess as prometheus_multiprocess
from PIL import Image
from flask import (
    Flask, Response, render_template, request, send_file, session, jsonify
)
from flask_apidoc import ApiDoc

from data import Storage, pil_to_dataURI
from face_detect import detect_faces
from metrics import (
    METRIC_INDEX_TIME,
    METRIC_PICK_RANDOM_TIME,
    METRIC_ADD_IMAGE_TIME,
    METRIC_APPLY_MASK_TIME,
    apply_inpainter
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from inpaint import make_inpainter

STATIC_DIR = 'static'
RANDOM_IMAGES_DIR = None
DEFAULT_RANDOM_IMAGES_DIR = os.path.join(STATIC_DIR, 'random_images')

INPAINTER = None

app = Flask(__name__)
app.secret_key = 'super secret key 228'
doc = ApiDoc(app=app)

def setup_app(
        model_state_dict=None,
        random_images_dir=DEFAULT_RANDOM_IMAGES_DIR,
        device='cpu'
):
    """Configure Flask application.
    
    Parameters
    ----------
    model_state_dict : str
        Path to a pytorch state dict compatible with the FaceRestore model.

    random_images_dir : str
        Path to a dir containing sample images

    device : str
        Device compatible with torch.device. Determines the device where
        an instance of the FaceRestore model will run.
        Examples: 'cpu', 'gpu', 'gpu:1'

    Returns
    -------
    app : flask.Flask
    """
    global INPAINTER
    global RANDOM_IMAGES_DIR
    INPAINTER = make_inpainter(model_state_dict)
    INPAINTER.set_device(device)
    RANDOM_IMAGES_DIR = random_images_dir
    return app


@app.route('/')
@METRIC_INDEX_TIME.time()
def index():
    return render_template('index.html')


@app.route('/pick_random')
@METRIC_PICK_RANDOM_TIME.time()
def pick_random():
    """
    @api {get} /pick_random Pick random photo
    @apiVersion 0.0.1
    @apiName pick_random
    @apiGroup FaceRestore

    @apiSuccess {File}     File      random image png
    """
    images = os.listdir(RANDOM_IMAGES_DIR)
    img_name = random.choice(images)
    return send_file(os.path.join(RANDOM_IMAGES_DIR, img_name))


@app.route('/add_image', methods=['POST'])
@METRIC_ADD_IMAGE_TIME.time()
def add_image():
    """
    @api {post} /add_image Add image
    @apiVersion 0.0.1
    @apiName add_image
    @apiGroup FaceRestore

    @apiParam {File}       image     Png file 256x256

    @apiSuccess {String}   image_id  Image Id in DB
    """
    image = Image.open(BytesIO(request.files['image'].read())).convert('RGB')
    image = image.resize((256, 256))

    image_id = Storage().save_image(image)
    session['image_id'] = str(image_id)
    return jsonify({
        'image_id': image_id
    })


@app.route('/apply_mask', methods=['POST'])
@METRIC_APPLY_MASK_TIME.time()
def apply_mask():
    """
    @api {post} /apply_mask Apply mask
    @apiVersion 0.0.1
    @apiName apply_mask
    @apiGroup FaceRestore

    @apiParam {String}     image_id  Id of source image to apply
    @apiParam {Integer}    step_id   Stroke number in history
    @apiParam {File}       mask      Png file 256x256, where painted is white and remained is black/transparent

    @apiSuccess {String}   image_id  Id of applied source image (unchanged)
    @apiSuccess {Integer}  step_id   Stroke number in history (unchanged)
    @apiSuccess {String}   result    DataURI of result png
    """
    storage = Storage()

    image_id = request.form['image_id']  # TODO: this is dangerous, but simplier for API
    step_id = int(request.form['step_id'])

    image = storage.get_image_by_id(image_id).convert('RGB')
    image = np.array(image, dtype=np.float32).transpose((2, 0, 1))

    painted_mask = Image.open(BytesIO(request.files['mask'].read()))
    painted_mask = painted_mask.resize(image.shape[1:][::-1]).convert('RGBA')
    painted_mask = np.array(painted_mask, dtype=np.float32)
    painted_pixels = (
        (painted_mask[:, :, 3] > 0) &
        (painted_mask[:, :, :3].max(axis=2) > 0)
    )
    mask = np.where(painted_pixels, 0, 1).astype(np.float32)
    mask = np.repeat(mask[None, :, :], 3, axis=0)
    mask_coverage = 1.0 - float(mask[0].mean())

    result = apply_inpainter(INPAINTER, mask, image)
    result = Image.fromarray((result.transpose((1, 2, 0)) * 255.).astype(np.uint8))

    mask_image = Image.fromarray((mask.transpose((1, 2, 0)) * 255).astype(np.uint8))
    storage.save_mask_and_result(image_id, step_id, mask_image, result)
    return jsonify({
        'image_id': image_id,
        'step_id': step_id,
        'result': pil_to_dataURI(result),
        'mask_coverage': mask_coverage
    })


@app.route('/detect_face', methods=['POST'])
def detect_face():
    image = Image.open(BytesIO(request.files['image'].read())).convert('RGB')
    image = image.resize((256, 256))
    faces = detect_faces(np.array(image))
    return jsonify({'faces': faces})


@app.route('/metrics')
def metrics():
    registry = prometheus.CollectorRegistry()
    prometheus_multiprocess.MultiProcessCollector(registry)
    return Response(
        prometheus.generate_latest(registry),
        mimetype=prometheus.CONTENT_TYPE_LATEST
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-state-dict')
    parser.add_argument(
        '-r', '--random-images-dir', default=DEFAULT_RANDOM_IMAGES_DIR
    )
    parser.add_argument('-d', '--device', default='cpu')
    args = parser.parse_args()
    setup_app(
        args.model_state_dict, args.random_images_dir, args.device
    ).run()


if __name__ == '__main__':
    main()
