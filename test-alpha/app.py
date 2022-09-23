from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np

# sub-imports
# from object_detection import Detect

app=Flask(__name__)

class Detect:

    def __init__(self, video_source,
                        classes,
                        config,
                        weights,
                        frame_title,
                        wait_key,
                        threshold,
                        suppression_threshold,
                        yolo_image_size):


                        self.video_source = video_source
                        self.classes = classes
                        self.config = config
                        self. weights = weights
                        self.frame_title = frame_title
                        self.wait_key = wait_key
                        self.threshold = threshold
                        self.suppression_threshold = suppression_threshold
                        self.yolo_image_size = yolo_image_size
                        self.detect_count = 0

    def find_objects(self, model_outputs, YOLO_IMAGE_SIZE, THRESHOLD, SUPPRESSION_THRESHOLD):
        bounding_box_locations = []
        class_ids = []
        confidence_values = []

        for output in model_outputs:
            for prediction in output:
                class_probabilities = prediction[5:]
                class_id = np.argmax(class_probabilities)
                confidence = class_probabilities[class_id]

                if confidence > THRESHOLD:
                    w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                    # the center of the bounding box (we should transform these values)
                    x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                    bounding_box_locations.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidence_values.append(float(confidence))

        box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

        return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values

    def mark_detected_objects(self, img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio,
                         height_ratio):
        for index in bounding_box_ids:
            bounding_box = all_bounding_boxes[index]
            x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
            
            # we have to transform the locations and coordinates because the image is resized

            x = int(x * width_ratio)
            y = int(y * height_ratio)
            w = int(w * width_ratio)
            h = int(h * height_ratio)

            # OpenCV deals with BGR blue green red (255,0,0) then it is the blue color
            # we are not going to detect every objects just PERSON and CAR
            # if class_ids[index] == 2:
            #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #     class_with_confidence = 'CAR' + str(int(confidence_values[index] * 100)) + '%'
            #     cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

            if class_ids[index] == 0:

                self.detect_count += 1

                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                class_with_confidence = f'drowning' + str(int(confidence_values[index] * 100)) + '%'
                cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)

# find_objects
# mark_detected_objects

    def generate_frames(self):
        capture = cv2.VideoCapture(self.video_source)
        
        neural_network = cv2.dnn.readNetFromDarknet(self.config, self.weights)

        neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        YOLO_IMAGE_SIZE = self.yolo_image_size

        while True:
            frame_grabbed, frame = capture.read()

            if not frame_grabbed:
                break
            else:

                original_width, original_height = frame.shape[1], frame.shape[0]

                # the image into a BLOB [0-1] RGB - BGR
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
                neural_network.setInput(blob)

                layer_names = neural_network.getLayerNames()
                # YOLO network has 3 output layer - note: these indexes are starting with 1
                output_names = [layer_names[index - 1] for index in neural_network.getUnconnectedOutLayers()]

                self.detect_count = 0

                outputs = neural_network.forward(output_names)
                predicted_objects, bbox_locations, class_label_ids, conf_values = self.find_objects(outputs,
                                                                                                self.yolo_image_size,
                                                                                                self.threshold,
                                                                                                self.suppression_threshold)

                self.mark_detected_objects(frame, predicted_objects, bbox_locations, class_label_ids, conf_values,
                                original_width / YOLO_IMAGE_SIZE, original_height / YOLO_IMAGE_SIZE)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()


            yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# global declaration
source = Detect(video_source = './media/swimming_pool1.mp4',
                    classes = ['drowning'],
                    config = './config/yolov3_testing.cfg',
                    weights = './weights/yolov3_training_3000.weights',
                    frame_title = 'YOLO V3 Object Detection',
                    wait_key = 10,
                    threshold = 0.5,
                    suppression_threshold = 0.4,
                    yolo_image_size = 320)

@app.route('/counter', methods=['POST'])
def counter():
    return jsonify('', render_template('counter.html', dyn_var = source.detect_count))

@app.route('/video')
def video():

    frame = source.generate_frames()

    return Response(frame, 
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', dyn_var = source.detect_count)

if __name__ == "__main__":
    app.run(debug=True)