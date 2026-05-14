import numpy as np
import prometheus_client as prometheus


# Metrics naming is inspired by this article:
# https://prometheus.io/docs/practices/naming
METRIC_REQUEST_TIME = prometheus.Summary(
    'request_seconds', 'Response time', ['route']
)
METRIC_INDEX_TIME = METRIC_REQUEST_TIME.labels('/')
METRIC_PICK_RANDOM_TIME = METRIC_REQUEST_TIME.labels('/pick_random')
METRIC_ADD_IMAGE_TIME = METRIC_REQUEST_TIME.labels('/add_image')
METRIC_APPLY_MASK_TIME = METRIC_REQUEST_TIME.labels('/apply_mask')

METRIC_APPLY_INPAINTER_TIME = prometheus.Summary(
    'face_restore_apply_seconds', 'Execution time of face restoration'
)


@METRIC_APPLY_INPAINTER_TIME.time()
def apply_inpainter(inpainter, mask, image):
    return inpainter(np.where(mask, image, 0) / 255., mask)
