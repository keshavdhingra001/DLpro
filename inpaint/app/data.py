import base64
import io
from datetime import datetime
from io import BytesIO

import pymongo
from PIL import Image
from bson import ObjectId


def pil_to_dataURI(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode('utf8')


class Storage:
    def __init__(self):
        conn = pymongo.MongoClient(host='mongo')
        db = conn.facerestore
        self.coll_images = db.images
        self.coll_masks = db.masks

    def get_image_by_id(self, image_id):
        dataURI_image = list(self.coll_images.find({'_id': ObjectId(image_id)}))[0]['image']
        base64_image = dataURI_image[len('data:image/png;base64,'):]
        return Image.open(BytesIO(base64.b64decode(base64_image)))

    def save_image(self, image: Image):
        result = self.coll_images.insert_one({
            "datatime": datetime.now(),
            "image": pil_to_dataURI(image)
        })
        return str(result.inserted_id)

    def save_mask_and_result(self, image_id, step_id, mask: Image, result: Image):
        result_doc = self.coll_masks.insert_one({
            "datatime": datetime.now(),
            "image_id": ObjectId(image_id),
            "step_id": step_id,
            "mask": pil_to_dataURI(mask),
            "result": pil_to_dataURI(result)
        })
        return str(result_doc.inserted_id)
