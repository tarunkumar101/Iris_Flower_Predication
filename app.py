from flask import Flask, render_template, request
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the Iris dataset
iris = load_iris()

# Train the classifier if the pickle file doesn't exist
try:
    with open('iris_classifier.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)
except FileNotFoundError:
    # If the pickle file doesn't exist, train a new classifier
    X, y = iris.data, iris.target
    classifier = RandomForestClassifier()
    classifier.fit(X, y)
    with open('iris_classifier.pkl', 'wb') as model_file:
        pickle.dump(classifier, model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    features = [float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])]

    # Make predictions using the loaded model
    prediction = classifier.predict([features])[0]
    iris_species = iris.target_names[prediction]

    # Format the prediction
    result = f'Predicted Iris Species: {iris_species}'

    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
