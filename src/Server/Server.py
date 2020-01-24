from flask import Flask , jsonify , request
import os
import base64
import datetime
from NetFunctions import ClassifyImage


app = Flask(__name__)


@app.route('/', methods=['POST'])  # Accept HTTP POST requests only
def handle_request():
    # Decode the image string
    imageString = base64.b64decode(request.form['image'])

    # Save the image with the current date as the name
    filepath = "C:/Users/shane/Pictures/MobileUploads/"
    filename = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    filepathname = filepath + filename + '.jpg'
    print(filepathname)
    with open(filepathname, 'wb') as f:
        f.write(imageString)

    # execute the classification script
    classify, name, sci_name, edible, url, probability = test.my_image(filepathname)

    # Append result to the image name
    os.rename(filepathname, (filepath + name + '_' + filename + '.jpg').replace(' ', '_'))

    return jsonify({"response": name, "sci": sci_name, "edible": edible, "url": url, "probability": probability})


test = ClassifyImage()
app.run(host="0.0.0.0", port=5000, debug=False)