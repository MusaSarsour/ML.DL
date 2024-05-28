from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib

app = Flask(__name__)

# Load the pretrained model
model = joblib.load('improved_stock_model.pkl')

@app.route('/')
def home():
    return render_template_string('''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <title>Stock Pricing</title>
        <style>
          body { background-color: #121212; color: #ffffff; font-family: Arial, sans-serif; }
          .container { display: flex; justify-content: center; align-items: center; height: 100vh; flex-direction: column; }
          .form-group { margin-bottom: 15px; }
          input, select { width: 100%; padding: 10px; margin: 5px 0 10px 0; border: none; background: #333333; color: #ffffff; }
          input[type="submit"] { background-color: #6200ea; color: #ffffff; cursor: pointer; }
          input[type="submit"]:hover { background-color: #3700b3; }
          h1 { margin-bottom: 20px; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Stock Pricing</h1>
          <form action="/predict" method="post">
            <div class="form-group">
              <label for="adjHigh">Adjusted High:</label>
              <input type="number" step="0.01" id="adjHigh" name="adjHigh" placeholder="Enter value between 100 and 200">
            </div>
            <div class="form-group">
              <label for="adjLow">Adjusted Low:</label>
              <input type="number" step="0.01" id="adjLow" name="adjLow" placeholder="Enter value between 90 and 190">
            </div>
            <div class="form-group">
              <label for="adjOpen">Adjusted Open:</label>
              <input type="number" step="0.01" id="adjOpen" name="adjOpen" placeholder="Enter value between 95 and 195">
            </div>
            <div class="form-group">
              <label for="adjVolume">Adjusted Volume:</label>
              <input type="number" step="1" id="adjVolume" name="adjVolume" placeholder="Enter value between 1000000 and 50000000">
            </div>
            <input type="submit" value="Predict">
          </form>
        </div>
      </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        adjHigh = float(request.form['adjHigh'])
        adjLow = float(request.form['adjLow'])
        adjOpen = float(request.form['adjOpen'])
        adjVolume = float(request.form['adjVolume'])

        # Validate input values
        if not (100 <= adjHigh <= 200):
            return jsonify(error="Adjusted High value should be between 100 and 200"), 400
        if not (90 <= adjLow <= 190):
            return jsonify(error="Adjusted Low value should be between 90 and 190"), 400
        if not (95 <= adjOpen <= 195):
            return jsonify(error="Adjusted Open value should be between 95 and 195"), 400
        if not (1000000 <= adjVolume <= 50000000):
            return jsonify(error="Adjusted Volume value should be between 1,000,000 and 50,000,000"), 400

        # Predict using the model
        features = np.array([[adjHigh, adjLow, adjOpen, adjVolume]])
        prediction = model.predict(features)
        return jsonify(prediction=prediction[0])
    except ValueError as e:
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
