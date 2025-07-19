

## 🔍 Fake News Detection on Kaggle

Kaggle hosts numerous datasets and notebooks dedicated to fake news detection. Here are some notable resources:

### 📊 Datasets

* **Fake News Detection Datasets**: This collection includes articles labeled as fake and real news, sourced from various platforms. It's suitable for training and evaluating machine learning models. ([Kaggle][1])([Kaggle][1])

* **Fake News Detection by Bhavik Jikadara**: A dataset aimed at analyzing news content to determine its truthfulness, facilitating text classification tasks. ([Kaggle][2])([Kaggle][2])

* **Fake News Classification (WELFake)**: A large dataset comprising over 70,000 news articles, categorized as real or fake, ideal for training deep learning models. ([Kaggle][3])([Kaggle][3])

### 🧠 Notebooks

* **Fake News Detection with LSTM**: This notebook demonstrates the use of Long Short-Term Memory (LSTM) networks for classifying fake news articles, achieving high accuracy. ([Kaggle][4])

* **NLP on Fake News with 99%+ Accuracy**: A notebook showcasing Natural Language Processing (NLP) techniques to classify fake news, achieving over 99% accuracy. ([Kaggle][5])

* **Fake News Detection with Transformers and TF-IDF**: This notebook combines Transformer models with Term Frequency-Inverse Document Frequency (TF-IDF) features for effective fake news detection. ([Kaggle][6])

---

## 📚 Academic Research

For a deeper understanding of fake news detection methodologies, consider exploring the following research papers:

* **3HAN: A Deep Neural Network for Fake News Detection**: Introduces a hierarchical attention network that processes news articles at the word, sentence, and headline levels, achieving 96.77% accuracy. ([arXiv][7])

* **A Benchmark Study of Machine Learning Models for Online Fake News Detection**: Compares various machine learning models, highlighting the effectiveness of BERT and similar pre-trained models for fake news detection. ([arXiv][8])

* **"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection**: Presents a large, manually labeled dataset for fake news detection, facilitating the development and evaluation of detection models. ([arXiv][9])

---

If you have specific questions or need assistance with a particular aspect of fake news detection, feel free to ask!

[1]: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?utm_source=chatgpt.com "Fake News Detection Datasets - Kaggle"
[2]: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection?utm_source=chatgpt.com "Fake News Detection - Kaggle"
[3]: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification?utm_source=chatgpt.com "Fake News Classification - Kaggle"
[4]: https://www.kaggle.com/code/lonnieqin/fake-news-detection-with-lstm-99-8-accuracy?utm_source=chatgpt.com "Fake News Detection with LSTM[99.8% accuracy] - Kaggle"
[5]: https://www.kaggle.com/code/swarnabh31/nlp-on-fake-news-99-accuracy?utm_source=chatgpt.com "NLP on Fake News 99%+ accuracy - Kaggle"
[6]: https://www.kaggle.com/code/jacopoferretti/fake-news-detection-with-transformers-and-tf-idf?utm_source=chatgpt.com "Fake News Detection with Transformers and TF-IDF - Kaggle"
[7]: https://arxiv.org/abs/2306.12014?utm_source=chatgpt.com "3HAN: A Deep Neural Network for Fake News Detection"
[8]: https://arxiv.org/abs/1905.04749?utm_source=chatgpt.com "A Benchmark Study of Machine Learning Models for Online Fake News Detection"
[9]: https://arxiv.org/abs/1705.00648?utm_source=chatgpt.com "\"Liar, Liar Pants on Fire\": A New Benchmark Dataset for Fake News Detection"





# 🚚 Delivery Time Estimation using Machine Learning

This project uses machine learning techniques to estimate the delivery time of food orders based on multiple inputs such as the delivery person's age, rating, vehicle type, order type, and geographic distance between restaurant and customer.

---

## 📘 Project Overview

The objective of this project is to build a machine learning model that can predict the estimated time required for a food delivery, based on structured data inputs. It involves:

* Data preprocessing
* Feature engineering
* Model training using Keras
* Deployment-ready preprocessing and inference pipeline

---

## 📂 Dataset Description

Features used in the model:

* `Delivery_person_Age`: Age of the delivery person
* `Delivery_person_Ratings`: Average customer rating
* `Type_of_order`: e.g., Snack, Meal, Beverages
* `Type_of_vehicle`: e.g., Bike, Scooter
* `Restaurant_latitude` and `Restaurant_longitude`: Coordinates of the restaurant
* `Delivery_location_latitude` and `Delivery_location_longitude`: Coordinates of the delivery location

Derived feature:

* `distance`: Calculated using the **Spherical Law of Cosines**

---

## 🛠️ Data Preprocessing Pipeline

To ensure consistent preprocessing during training and inference, a Scikit-learn pipeline is created and saved.

### 🔹 Features:

* **Numerical**:

  * `Delivery_person_Age`
  * `Delivery_person_Ratings`
  * `distance`
* **Categorical**:

  * `Type_of_order`
  * `Type_of_vehicle`

### 🔧 Transformations:

* Numerical features are scaled using `MinMaxScaler`
* Categorical features are one-hot encoded using `OneHotEncoder` with `handle_unknown='ignore'`

### 💾 Pipeline Code:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance']
categorical_features = ['Type_of_order', 'Type_of_vehicle']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

pipeline.fit(X)  # Replace X with your input features DataFrame
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')
```

---

## 🧠 Model Details

A deep learning model was built using Keras (`model-final.h5`) to predict scaled delivery time. It was trained using preprocessed input data, and the output was inverse transformed using a saved `target_scaler.pkl`.

---

## 📦 Files Included

* `model-final.h5`: Trained Keras model
* `preprocessing_pipeline.pkl`: Scikit-learn pipeline with MinMaxScaler and OneHotEncoder
* `target_scaler.pkl`: Scaler for the model's output
* `app.py`: Flask application for deployment
* `index.html`: Frontend interface for input and prediction

---

## 🚀 How to Run

1. Clone this repo or copy files locally.
2. Install required packages:

   ```bash
   pip install flask numpy pandas scikit-learn tensorflow joblib
   ```
3. Run the Flask app:

   ```bash
   python app.py
   ```
4. Open in browser: `http://127.0.0.1:5000`

---

## 🧪 Sample Prediction Flow

1. User submits form data (age, rating, coordinates, order/vehicle types).
2. Distance is calculated using geographic coordinates.
3. Data is preprocessed using the saved pipeline.
4. Preprocessed data is passed into the Keras model.
5. Output is inverse-scaled and returned as estimated delivery time.

---

## 📌 Future Improvements

* Add support for time-related features (hour of the day, day of week)
* Use real route distance from APIs (e.g., Google Maps)
* Extend to handle live data and integrate SMS/alerts
* Convert to a mobile-friendly responsive app

---

## 👤 Author

**Dhavasig Dhavasi**
🔗 [Kaggle Profile](https://www.kaggle.com/dhavasigdhavasi)

---

