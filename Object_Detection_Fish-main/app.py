import base64
import cv2
import math
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

def image_detection(image_path):
    # Load the image
    video_capture = image_path
    cap=cv2.VideoCapture(video_capture)    
    success, img_real = cap.read()

    # Load the YOLOv8 model
    model = YOLO('best.pt')
    
    # Define the class names
    classNames = ["Columnaris Disease", "EUS Disease", "Gill Disease", "Healthy Fish", "Streptococcus Disease"]
        
    target_size=(128, 128)
    
    with Image.open(image_path) as s:
        width, height = s.size
    
    original_size=(width, height)
    
    # Resize the image
    img_resized = cv2.resize(img_real, target_size)
    
    # Perform detection
    results = model(img_resized, conf=0.7, stream=True)
    label = 'No detections'
    class_name = label
    conf = ''
    # Process each detection
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (255, 0, 255), 3)
            conf = math.ceil((box.conf[0] * 100))
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}%'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img_resized, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img_resized, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    # Resize the image back to its original size
    img_original = cv2.resize(img_resized, original_size, interpolation=cv2.INTER_LANCZOS4) if original_size else img_resized
    
    # Encode image to base64
    _, img_encoded = cv2.imencode('.jpg', img_original)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Return the base64-encoded image and label
    if conf == '':
        return img_base64, label
    else:
        return img_base64, class_name, conf

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detectApi', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image temporarily
    image_path = 'uploaded_image.jpg'
    image.save(image_path)

    conf = 0
    # Perform detection on the uploaded image
    try:
        img_base64, label, conf = image_detection(image_path)
    except:
        img_base64, label = image_detection(image_path)
   
    # Convert base64 string to bytes
    #img_bytes = base64.b64decode(img_base64)
    # Save the bytes to a file  
    #output_file_path = r'C:\Users\DATA Technology\Desktop\new.jpg'
    #with open(output_file_path, 'wb') as f:
    #    f.write(img_bytes)

    # Return the base64-encoded image and label as JSON
    if conf == 0:
        return jsonify({'image': img_base64, 'Detection': label})
    else:
        return jsonify({'image': img_base64, 'Detection': label, 'Percentage': conf})


if __name__ == '__main__':
    app.run(debug=True)
