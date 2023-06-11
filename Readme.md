**Project Title: Polyglot Interpreter**

**Project Description:**

Hand-filled hotel feedback forms accumulate daily, overwhelming staff and managers. Consequently, these valuable customer insights often go unexplored, as they are either shelved or discarded due to time constraints. The task at hand is the automatic detection and extraction of hand marked heterogeneous checkboxes (e.g. checkbox, radio button, Yes/No, numeric rating) and multilingual handwritten text (e.g. English, Urdu). Digitization of handwritten forms will allow hotels to gain meaning insights from customer feedback and improve their services. 
To digitize hotel feedback forms, we need to extract user credentials, user ratings e.g. checkboxes, radio buttons, numeric rating and user comments that can be written in Urdu and English language both. To extract user ratings, we have proposed efficient object detection methodologies. For extraction of handwritten text, we have proposed lightweight custom OCR models for English and Urdu language. 
Once these methodologies are combined we are able to digitize the complete form and store it in a database. We can run multiple queries and apply different analytic techniques on the stored data. The analysis will be viewed on a dashboard for hotel employees to gain insights and improve customer services.  

**Flow of Project:**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/cd480564-df15-4107-847e-397f8c5a6243)


**High Level Architecture of the project:**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/16a4a472-3a92-4c7f-b8ef-7cb3a66c1116)


The Polyglot Interpreter is a comprehensive solution for digitizing hotel feedback forms, handling both structured and unstructured data, such as multiple-choice questions and handwritten text. A robust dataset was created to train the system, generating four subsets to focus on different aspects of the digitization process: detection and localization of data, line segmentation of text, Optical Character Recognition (OCR) for handwriting, and extraction of numeric data.

Different models were trained for each stage of the process, with the YOLO v5 model used for data localization and line segmentation, and custom OCR models for handwritten text conversion. User credentials were digitized with TrOCR, while multiple-choice questions were segmented and digitized using EasyOCR. Handwritten comments were processed with a combination of TrOCR and UrduLiteOCR.

EasyOCR was chosen over TrOCR for digitizing the questions due to its speed and ease of handling multi-line text. All digitized data was stored in a dedicated database for future analysis and insights. The Polyglot Interpreter therefore represents an innovative solution for transforming analogue customer feedback into actionable, digital insights in the hospitality industry.


**Project Expected Outcome:**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/aadf0843-7c0b-4412-8fe8-4445051c0feb)


**Goals of MLOps Project:**


The goals of this project are to create a Mlops pipeline and containerization for Polyglot interpreter. The major reasons being, so that us students can get hands on practice of how to implement complex MLops pipelines. The second reason being so that this application be delivered and shared with other people without any hassle. We are going to containerize this application and make it deliverable to other people. Another reason is that in future if the model starts to degrade, or we come across new data samples we can easily retrain and deploy the model without going through the hassle of training each model, creating datasets and loading them or orchestrating the whole pipeline hassle free. 


**Data versioning**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/95c3934a-de9f-4794-a5f6-d9746170ef5b)

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/550bbe96-882a-4b4b-94fb-453760b58745)


**Training YOLO models**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/784d4f79-4b2e-4391-8580-cea0ead4ece4)


![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/918274bd-ef60-492e-b9ce-ed93e9214a44)


**Training OCR mode**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/0dfb4ba1-603f-4425-923d-719eccd01b91)


**Creating an inference\prediction app**

![image](https://github.com/abdullah1772/MLOps_Project/assets/88187437/cd1d8c14-987f-4a4d-afdd-85689a07f885)


**Containerizing and making the app delivery ready**

![jenkins01](https://github.com/abdullah1772/MLOps_Project/assets/88187437/c4c38121-aed8-48cf-a1b7-bd60289c2885)


![jenkins02](https://github.com/abdullah1772/MLOps_Project/assets/88187437/2c015f74-7a70-4b39-9078-47ed650f60f6)


![jenkins03](https://github.com/abdullah1772/MLOps_Project/assets/88187437/f6824b80-112f-4832-840d-0ee9188c0194)


![Docker_app](https://github.com/abdullah1772/MLOps_Project/assets/88187437/546a1b68-edf0-4263-b977-5068f8947bbb)



**How to use**
Getting Started

To clone and run this application, you'll need Git installed on your computer. From your command line:
Step 1: Clone the Repo

Clone this repository using the following command:


git clone https://github.com/abdullah1772/MLOps_Project.git

Navigate to the project directory:


cd MLOps_Project

python install -r reqirements.txt

Step 2: Clone YOLOv5 into the Project Directory

In the MLOps_Project folder, clone the YOLOv5 repository:


git clone https://github.com/ultralytics/yolov5.git


Step 3: Run the Apache Airflow Script

Run the YOLO_Training_Apache_DAGS.py script using Apache Airflow. Be sure to replace 'path/to' with the path to your dataset:

Replace 'path/to' with your actual dataset path in the file YOLO_Training_Apache_DAGS.py 

Run the DAGS from apache airflow server

Step 4: For retraining the OCR model
 
1.	Run the UrduOCR.py using MLflow
2.	Don't forget to change the path to your dataset and save model location

Step 5: Inference app
1.To run the inference app provide the path in the Kl.py to your model weights 
2.Run Streamlit run Kl.py

Step 6: Docker File

 



**Learning Outcomes:**
1.	Data versioning using dvc
2.	Large data sharing using DVC
3.	Training multiple YOLO models using Airflow and scheduling them
4.	Integrating Wandb with YOLO models in Airflow
5.	Using MLflow for model logging and sharing artifacts.
6.	Sharing and Versioning model weights using DVC
7.	Creating Inference app 
8.	Creating CI/CD pipeline for multiple machine learning models
9.	Containerizing application and making it useable without dependency issues
10.	Automate the process of training and testing models on different machine with out dependency issues 

Contributing to MLOps_Project

To contribute to MLOps_Project, follow these steps:

    Fork the repository.
    Create a new branch: git checkout -b <branch_name>.
    Make your changes and commit them: git commit -m '<commit_message>'
    Push to the original branch: git push origin <project_name>/<location>

