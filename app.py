from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, disconnect
import base64
import numpy as np
import cv2
from flask_httpauth import HTTPDigestAuth
import os
from dotenv import load_dotenv
from engineio.payload import Payload
from queue import Queue, Empty
from custom_flask import CustomFlask

Payload.max_decode_packets = 500
load_dotenv(verbose=True)

image_queue = Queue(maxsize=50)
processed_queue = Queue(maxsize=50)

auth = HTTPDigestAuth()

def find_objects_yolo(yolo_model , img , colors , class_labels , img_height , img_width):
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layers = yolo_model.getUnconnectedOutLayersNames()
    blob_img = cv2.dnn.blobFromImage(img , scalefactor = 1/255 , size = (416 , 416) , swapRB = True , crop = False)
    yolo_model.setInput(blob_img)
    obj_detection_layers = yolo_model.forward(yolo_output_layers)
    class_ids_list = []
    boxes_list = []
    confidences_list = []
    for obj_det_layer in obj_detection_layers:
        for obj in obj_det_layer:
            scores = obj[5:]
            bounding_box = obj[0:4] * np.array([img_width , img_height , img_width , img_height])
            (box_centre_x_pt , box_centre_y_pt , box_width , box_height) = bounding_box.astype("int")
            start_x_pt = int(box_centre_x_pt - (box_width/2))
            start_y_pt = int(box_centre_y_pt - (box_height/2))
            predicted_class_id = np.argmax(scores)
            confidence_score = float(scores[predicted_class_id])
            if confidence_score > 0.40:
                class_ids_list.append(predicted_class_id)
                confidences_list.append(confidence_score)
                boxes_list.append([start_x_pt , start_y_pt , int(box_width) , int(box_height)])
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list , confidences_list , 0.5 , 0.4)
    output_img = img
    for max_value_id in max_value_ids:
        max_class_id = max_value_id[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]
        prediction_confidence = prediction_confidence*100
        text = predicted_class_label + ":" + str(int(prediction_confidence)) + "%"
        color_box = colors[predicted_class_id]
        color_box = (int(color_box[0]), int(color_box[1]), int(color_box[2])) 
        output_img = cv2.rectangle(output_img , (start_x_pt , start_y_pt) , (start_x_pt+box_width , start_y_pt+box_height) , color_box , thickness = 2)
        output_img = cv2.putText(output_img , text , (start_x_pt , start_y_pt-2) , cv2.FONT_HERSHEY_SIMPLEX , 0.6 , color = color_box , thickness = 1)
    return output_img


def _detect_mask(img):
	cfg = 'mask_yolov4.cfg'
	weight = 'mask_yolov4_best.weights'
	class_labels = ['without_mask' , 'with_mask']
	colors = [[0 , 0 , 255] , [0 , 255 , 0]]
	yolo_model = cv2.dnn_DetectionModel(cfg , weight)
	yolo_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	yolo_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	height = img.shape[0]
	width = img.shape[1]
	img = find_objects_yolo(yolo_model , img , colors , class_labels , img.shape[0] , img.shape[1])
	return img


def _base64_decode(img):
    _, buffer = cv2.imencode(".jpg", img)
    base64_data = base64.b64encode(buffer)
    base64_data = "data:image/jpg;base64," + base64_data.decode('utf-8')
    return base64_data


def _base64_encode(img_base64):
    img_binary = base64.b64decode(img_base64)
    jpg = np.frombuffer(img_binary, dtype=np.uint8)
    img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    return img


def _validate_access_token():
    access_token = request.headers.environ.get('HTTP_X_ACCESS_TOKEN')
    if access_token != os.environ.get("ACCESS_TOKEN"):
        disconnect()


def loop_emit():
    print("start loop")
    while True:
        try:
            img = image_queue.get()
        except Empty:
            continue

        processed_img = _detect_mask(img)
        base64_data = _base64_decode(processed_img)
        processed_queue.put(base64_data)


app = CustomFlask(__name__, background_task=loop_emit)
app.config['SECRET_KEY'] = os.environ.get("APP_SECRET")
socketio = SocketIO(app, cors_allowed_origins="*")


@auth.get_password
def get_pw(username):
    if username == os.environ.get("USER_NAME"):
        return os.environ.get("PASSWORD")
    return None


@app.route('/health_check')
def health_check():
    return "Status OK"


@app.route('/sender')
@auth.login_required
def sender():
    return render_template("sender.html", access_token=os.environ.get("ACCESS_TOKEN"))


@app.route('/receiver')
@auth.login_required
def receiver():
    return render_template("receiver.html", access_token=os.environ.get("ACCESS_TOKEN"))


@socketio.on('connect', namespace="/image")
def test_connect():
    _validate_access_token()

    referer = request.referrer

    if referer is None or 'receiver' not in referer:
        image_queue.queue.clear()
        processed_queue.queue.clear()


@socketio.on("send image", namespace="/image")
def parse_image(json):
    _validate_access_token()

    img_base64 = json["data"].split(',')[1]
    img = _base64_encode(img_base64)
    image_queue.put(img)

    try:
        base64_data = processed_queue.get()
    except Empty:
        return
    else:
        emit('return img', base64_data, broadcast=True)


if __name__ == '__main__':
    socketio.run(app, debug=False)