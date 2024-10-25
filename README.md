**🧠 Advanced Brain Stroke Detection and Prediction System 🧠 : Integrating 3D Convolutional Neural Networks and Machine Learning on CT Scans and Clinical Data**

<img width="379" alt="image" src="https://github.com/user-attachments/assets/713bd0ea-bddf-4eb0-9d96-29ad231968de">


Welcome to our Advanced Brain Stroke Detection and Prediction System! This project combines the power of Deep Learning and Machine Learning to provide an innovative approach to stroke diagnosis and risk prediction, leveraging 3D CNNs for CT scan analysis and a predictive ML pipeline based on clinical data. 🚀💉

📜 Abstract: Stroke is a leading cause of severe long-term disability and mortality worldwide. Timely detection and accurate risk prediction are crucial to reduce its impact on individuals and healthcare systems. This project introduces an Advanced Brain Stroke Detection and Prediction System that utilizes 3D Convolutional Neural Networks (3D CNNs) for stroke identification from CT scans, combined with Machine Learning (ML) models for predicting stroke risk based on clinical data. The 3D CNN model performs image analysis on volumetric CT data to classify brain images into stroke and non-stroke categories. Additionally, clinical risk factors, such as age, lifestyle, and comorbidities, are analyzed using ML classifiers, achieving a prediction accuracy of 98% with XGBoost. This system provides an end-to-end solution for early stroke detection and risk assessment, supporting healthcare providers in decision-making processes.

🌟 Introduction

Stroke continues to be a global health concern, accounting for a significant number of disabilities and fatalities annually. While brain imaging modalities, such as CT scans, are essential for stroke diagnosis, interpreting these scans accurately and efficiently remains challenging. Machine Learning (ML) and Deep Learning (DL) advancements have created opportunities to enhance stroke detection and risk prediction. This project proposes a comprehensive solution that leverages 3D CNNs for analyzing CT scans to detect brain strokes with high accuracy. Additionally, ML algorithms, including Decision Trees, KNN, XGBoost, and Random Forest, are applied to clinical data to predict the likelihood of a stroke based on factors like age, hypertension, heart disease, and lifestyle choices. The hybrid approach combines both visual and clinical data for a multi-dimensional analysis, aiming to deliver improved diagnostic accuracy and timely predictions that aid clinicians and improve patient outcomes.

🚀 Proposed Methodology: This project methodology comprises two primary components:

1. Stroke Detection from CT Scans: Using 3D CNNs for analyzing brain scans.
   
2. Stroke Risk Prediction from Clinical Data: Using ML models for clinical feature-based prediction.
   
**(i). Stroke Detection from CT Scans Using 3D CNN**

Dataset: Brain Stroke CT Image Dataset: A dataset containing brain CT scans labeled as either "Normal" or "Stroke," sourced from Kaggle.

Preprocessing: Data Augmentation: To increase the robustness of the model, advanced data augmentation techniques are applied, including:

Rotation and Zooming: To simulate different angles and scales of the brain CT scans.

![image](https://github.com/user-attachments/assets/2df717cd-2601-4f25-8e32-5162b6b731a2)


Flipping: To account for anatomical variations.

![image](https://github.com/user-attachments/assets/9879b880-7e76-400a-a4e0-980d7e8295c8)


Normalization: Ensures consistent pixel values across all CT scans.

![image](https://github.com/user-attachments/assets/0d013781-d720-4d66-8e50-bc27be52b76d)


**Model Architecture: 3D CNN**: A Convolutional Neural Network adapted to process three-dimensional data, enabling it to learn spatial features across multiple frames of CT scans.

**Layers:Convolutional Layers**: Extract spatial features from volumetric data.

**Pooling Layers**: Downsample features to reduce dimensionality and computational load.

**Fully Connected Layers**: Generate the final prediction (Stroke or Normal) based on the extracted features.

**Training and Evaluation**

**Training**: The model is trained with extensive data augmentation to prevent overfitting.

**Evaluation**: Metrics include Accuracy, Precision, Recall, F1-score, and ROC AUC for performance assessment.

**(ii). Stroke Risk Prediction from Clinical Data Using ML Models**

Dataset: Stroke Prediction Dataset: A clinical dataset sourced from Kaggle containing patient data with features such as age, gender, hypertension, heart disease, and lifestyle factors.

**Preprocessing**

**Data Cleaning**

Handling NULL Values: Missing values are imputed based on the dataset characteristics.

Feature Encoding: Categorical features, such as gender, are one-hot encoded to be suitable for ML models.

Normalization: Standard scaling is applied to numerical features for uniformity.

Model Selection: Several machine learning algorithms were explored for predicting stroke risk

Decision Tree: A tree-based model that segments the data based on feature splits.

K-Nearest Neighbors (KNN): Uses proximity-based classification, assigning labels based on neighboring data points.

Random Forest: An ensemble of decision trees that reduces variance and improves accuracy.

XGBoost: An optimized gradient-boosted decision tree model, showing the highest accuracy in predictions.

**Training and Evaluation**

Training: Each ML model was trained and evaluated on the preprocessed clinical dataset.

Evaluation: The models were assessed based on Accuracy, Precision, Recall, F1-score, and ROC AUC. XGBoost emerged as the top-performing model with an accuracy of 98%.

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

🔮 Future Work

While this project demonstrates promising results in both stroke detection from CT scans and stroke risk prediction from clinical data, there are several potential improvements and research avenues that could further enhance its utility, accuracy, and applicability in real-world settings. Below are some future enhancements planned for this system:

1. 🌐 Integration of Multi-modal Data

This project currently utilizes CT imaging for stroke detection and clinical records for risk prediction. However, integrating additional modalities such as MRI scans or histopathological data could provide a more comprehensive understanding of stroke risk and progression. Future work could involve:

MRI Data Fusion: Combining CT and MRI scans to improve the accuracy of stroke localization and enhance model generalizability.
EHR Data Integration: Leveraging electronic health records (EHR) for more extensive patient data, which could provide richer context and potentially improve predictive performance.

2. 📈 Model Optimization and Transfer Learning

The current 3D CNN model is trained specifically on the CT image dataset. Applying transfer learning techniques could allow the model to generalize more effectively across different imaging datasets and improve performance:

Transfer Learning from Pre-trained Medical Models: Adapting existing pre-trained models for medical imaging, such as VGG, ResNet, or EfficientNet, could improve efficiency and accuracy.

Fine-tuning Hyperparameters: Conducting a thorough hyperparameter search to optimize the 3D CNN and ML models can help improve their predictive performance.
Cross-validation Across Datasets: Validating models on diverse datasets from different regions and demographics to enhance generalizability.

3. 🤖 Deployment as a Clinical Decision Support System (CDSS)
To bring the model into clinical practice, developing a user-friendly, web-based CDSS application for healthcare professionals is essential:

Real-time Prediction Interface: Developing a streamlined interface where clinicians can upload CT images and input clinical data for immediate stroke detection and risk assessment.

Integration with PACS Systems: Integrate with hospital Picture Archiving and Communication Systems (PACS) to automate CT image uploads and patient data fetching, simplifying workflow.

Explainable AI (XAI) Features: Adding explainable AI components could help medical practitioners understand the model's predictions, potentially increasing trust and reliability in the clinical setting.

4. 🧠 Enhanced Data Augmentation and Synthetic Data Generation
To address data limitations, advanced augmentation and synthetic data generation techniques can be applied:

Synthetic Image Generation: Utilizing GANs (Generative Adversarial Networks) to create synthetic CT images that simulate various stroke conditions, thereby enriching the training data.

Advanced Augmentation Techniques: Using augmentations that reflect realistic anatomical variations, such as random brain region shifts or contrast adjustments, to make the model more robust.

Federated Learning: Implementing federated learning approaches to train the model on data from multiple institutions without the need for direct data sharing, thereby enhancing data privacy and diversity.

5. 💻 AI Model Deployment and Edge Computing
For efficient and accessible deployment, especially in remote or resource-limited settings:

Edge Deployment: Deploy lightweight versions of the model on edge devices, such as smartphones or IoT devices, to enable stroke prediction and detection in regions with limited access to advanced computing infrastructure.

Quantization and Model Pruning: Applying model compression techniques like quantization and pruning can reduce computational overhead and enable faster inference on edge devices.

6. 🔍 Longitudinal and Predictive Analysis
The current model assesses stroke risk based on static data. However, incorporating a temporal component could allow for more dynamic risk assessments:

Time-Series Analysis: Use longitudinal data from multiple patient visits to predict stroke risk over time.

Progression Analysis: Develop a predictive model for stroke severity or progression based on the initial CT scan and follow-up imaging, aiding in treatment planning and early intervention.

7. 🛠 Incorporation of Explainable AI (XAI) Techniques
Explainability is crucial in medical applications to ensure clinicians trust and understand the AI system's decisions:

Grad-CAM for CT Images: Utilize Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight specific areas in CT images that influence the stroke detection model’s decision.

SHAP for Clinical Data: Implement SHAP (SHapley Additive exPlanations) values to provide detailed feature importance analysis for the ML-based stroke risk model, showing which clinical features impact each prediction.

💡 Conclusion
This project successfully demonstrates a combined approach for brain stroke detection and prediction, utilizing both CT scans and clinical data. The high accuracy achieved in both tasks illustrates the efficacy of integrating 3D CNNs with traditional ML models for medical diagnosis and risk assessment. Future enhancements include expanding the dataset, model optimization, and deploying a real-time web-based application for practical use.

📬 Contact
Feel free to reach out if you have any questions!

Email: adityakumar.singh2020a@vitstudent.ac.in

LinkedIn: https://www.linkedin.com/in/iamadi1709/

Thank you for your interest in this project! If you found this helpful, please give it a ⭐!
