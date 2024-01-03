from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import joblib
from skimage.feature import hog
import base64

app = Flask(__name__)

# Load the character recognition model
char_model = load_model('HCR_English.h5')

# Load the digit recognition model and preprocessor
clf, pp = joblib.load("digits_cls1.pkl")

char_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
    40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n',
    50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x',
    60: 'y', 61: 'z'
}

digit_dict = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_char', methods=['POST'])
def predict_char():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Character Recognition
        char_prediction, char_uploaded_image = recognize_characters(im)

        return render_template(
            'index.html',
            char_prediction=char_prediction,
            char_uploaded_image=char_uploaded_image,
            digit_prediction=None,
            uploaded_digits=None,
            digit_count=None,
            group_predictions=None,
            group_uploaded_images=None,
            group_count=None,
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Digit Recognition
        digit_predictions, uploaded_digits = recognize_digits(im)

        # Pass the length of digit predictions to the template
        digit_count = len(digit_predictions)

        return render_template(
            'index.html',
            char_prediction=None,
            char_uploaded_image=None,
            digit_prediction=digit_predictions,
            uploaded_digits=uploaded_digits,
            digit_count=digit_count,
            group_predictions=None,
            group_uploaded_images=None,
            group_count=None,
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/predict_group', methods=['POST'])
def predict_group():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    try:
        # Read the image
        im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Character Recognition for a group of characters
        group_predictions, group_uploaded_images = recognize_group_characters(im)

        return render_template(
            'index.html',
            char_prediction=None,
            char_uploaded_image=None,
            digit_prediction=None,
            uploaded_digits=None,
            digit_count=None,
            group_predictions=group_predictions,
            group_uploaded_images=group_uploaded_images,
            group_count=len(group_predictions),
            error=None
        )

    except Exception as e:
        return render_template('index.html', error=str(e))

def recognize_characters(im):
    # Convert the image to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Resize the image
    im_gray = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Threshold the image
    _, im_thresh = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Ensure the image has a single channel
    if len(im_thresh.shape) > 2:
        im_thresh = cv2.cvtColor(im_thresh, cv2.COLOR_BGR2GRAY)

    # Use the character model for prediction
    char_prediction = predict_character(im_thresh)

    # Convert the character image to base64 for display in HTML
    char_uploaded_image = base64.b64encode(cv2.imencode('.png', im_thresh)[1].tobytes()).decode()

    return char_prediction, char_uploaded_image

def predict_character(im):
    # Implement character prediction using the loaded character model (char_model)
    # Example: Replace this with the actual prediction logic
    char_probabilities = char_model.predict(np.expand_dims(im, axis=0))
    char_label = np.argmax(char_probabilities, axis=1)[0]
    char_prediction = char_dict.get(char_label, 'Unknown Character')
    return char_prediction

def recognize_digits(im):
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles containing each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    digit_predictions = []
    uploaded_digits = []

    for rect in rects:
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualize=False)
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        # Predict digit
        nbr = clf.predict(roi_hog_fd)
        digit_predictions.append(digit_dict.get(int(nbr[0]), 'Unknown Digit'))
        # Convert the digit image to base64 for display in HTML
        uploaded_digits.append(base64.b64encode(cv2.imencode('.png', roi)[1].tobytes()).decode())

    return digit_predictions, uploaded_digits

def recognize_group_characters(im):
    # Convert the image to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, im_thresh = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    contours, _ = cv2.findContours(im_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    group_predictions = []
    group_uploaded_images = []

    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the character region
        char_region = im_thresh[y:y + h, x:x + w]

        # Resize the character image
        char_region = cv2.resize(char_region, (28, 28), interpolation=cv2.INTER_AREA)

        # Use the character model for prediction
        char_prediction = predict_character(char_region)

        # Convert the character image to base64 for display in HTML
        char_uploaded_image = base64.b64encode(cv2.imencode('.png', char_region)[1].tobytes()).decode()

        group_predictions.append(char_prediction)
        group_uploaded_images.append(char_uploaded_image)

    return group_predictions, group_uploaded_images

if __name__ == '__main__':
    app.run(debug=True)
