<img width="526" alt="image" src="https://github.com/user-attachments/assets/478baedf-4f02-431e-a6a2-c34eaef5e2c8">🧠 Advanced Brain Stroke Detection and Prediction System 🧠 : Integrating 3D Convolutional Neural Networks and Machine Learning on CT Scans and Clinical Data

<img width="379" alt="image" src="https://github.com/user-attachments/assets/713bd0ea-bddf-4eb0-9d96-29ad231968de">

Welcome to our Advanced Brain Stroke Detection and Prediction System! This project combines the power of Deep Learning and Machine Learning to provide an innovative approach to stroke diagnosis and risk prediction, leveraging 3D CNNs for CT scan analysis and a predictive ML pipeline based on clinical data. 🚀💉

🌟 Features

🔍 3D CNN for Stroke Detection: Uses volumetric brain CT scans for better accuracy. The attached image shows volumetric CT scan data for brain stroke patients. This dataset contains numerous artifacts that require preprocessing before it can be used for further processing with a 3D CNN model.

<img width="1091" alt="image" src="https://github.com/user-attachments/assets/548f6bbe-bda2-405e-9903-54c2595967c6">

🩺 Stroke Risk Prediction: Estimates stroke likelihood based on clinical and lifestyle factors.

<img width="433" alt="image" src="https://github.com/user-attachments/assets/c6e88ef3-94c1-4aec-b900-0807c17338f7">

<img width="728" alt="image" src="https://github.com/user-attachments/assets/33b7845f-6ccd-4cdb-a6d4-9c4eb08cac26">

<img width="517" alt="image" src="https://github.com/user-attachments/assets/f0a0c378-b11f-4f86-8f98-13c6bdc3c914">

<img width="515" alt="image" src="https://github.com/user-attachments/assets/d208bde9-fd02-40e1-906a-55ca903c0dc3">

🎛 Advanced Data Augmentation: Improves model performance and tackles class imbalance.

<img width="530" alt="image" src="https://github.com/user-attachments/assets/35c8a608-ef9a-4e42-984e-b3a429d41dbc">

<img width="907" alt="image" src="https://github.com/user-attachments/assets/d073acc0-ecb7-44e8-b9e8-e759dde37326">

📊 High-Performance Models: Achieved 98% accuracy on XGBoost for stroke prediction!

![image](https://github.com/user-attachments/assets/2573c5ea-a18b-4c74-8474-f5298908037c)

📈 Comprehensive Evaluation: Metrics including Accuracy, Precision, Recall, F1-score, and ROC AUC.

![image](https://github.com/user-attachments/assets/af47474b-86c2-43df-8085-7f5f0501f9df)

📂 Dataset
Brain Stroke CT Image Dataset 🖼
Source: Kaggle : https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset
Description: Contains brain CT scans labeled as "Normal" and "Stroke".

Stroke Prediction Dataset 📄
Source: Kaggle : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
Description: Clinical and demographic data with health parameters for stroke risk analysis.

🚀 Quick Start
Data Preprocessing:

CT Scans: Preprocesses with denoising, resizing, and augmentation to enhance model training.
Prediction Data: Imputes missing values, handles categorical features, and normalizes data.

Model Training: Running train_3dcnn.py for 3D CNN stroke detection.

Run train_ml_models.py for stroke prediction using Decision Trees, KNN, XGBoost, and Random Forest.

Model Evaluation: ROC AUC Score, Confusion Matrix, Precision, Recall, and F1-score for each model.

🧪 Results
3D CNN Stroke Detection: Achieves high accuracy with advanced data augmentation techniques.

Stroke Prediction with XGBoost: 98% Accuracy and 99.81% ROC AUC. 🎉

Evaluation Metrics: ROC curves, confusion matrices, and more visualized in results/ 📊.

📈 Key Insights

CT Scan Challenges 🖼️: Augmentation methods like rotation, zooming, and flipping help overcome dataset limitations.

Clinical Data Analysis 📄: Imbalanced dataset handling, feature encoding, and data normalization were key to achieving high accuracy.

Model Comparison 🤖: XGBoost outperformed other models, thanks to its robustness with skewed data.

💡 Future Work

🧠 Expand Dataset: Add more CT scan data to improve model robustness.

🔄 Model Optimization: Experiment with ensemble learning to further boost prediction accuracy.

🌐 Web App Deployment: Deploy a user-friendly web app for real-time predictions.



