# MLOps-Loan-Eligibility-Pipeline/Deployment-on-GCP
load eligibility model deployed 

## Aim
To build optimal MLops pipeline on Google cloud platform to deploy loan eligibility
prediction model in production

## Tech Stack
* ➔ Language: Python
* ➔ Libraries: Flask, gunicorn, scipy, xgboost, joblib, seaborn, fancyimpute, scikit_learn
* ➔ Services: Flask, Docker, GCP, Gunicorn

## Prerequisites
It is advisable to have a basic knowledge of the following services to better understand the project.
* Flask
* Docker
* Cloud Build
* Cloud Run
* Cloud Source Repository

## Approach
### Step 1:
1. Clone the repository
2. Create a Flask App (app.py)
3. Build a Dockerfile
##### Once the files are created, create a new repository and commit the changes. From here on, this will be your source repository. Proceed with the below steps
### Step 2: Cloud Build Trigger
1. In your GCP console, create a new cloud build trigger
2. Point the trigger to your source repository
### Step 3: Cloud Run
1. In Cloud Run, point the CI/CD server towards your cloud build trigger out
2. The output from cloud build will be in Artifacts Registry, which holds a docker image
3. Cloud run will provide an endpoint, a HTTPS URL that will serve the flask app created
4. Add the permission "allUsers" with roles as "Cloud Run Invoker" and save the changes
5. Once changes the change reflects, the HTTPS URL will be accessible

### Project Structure
```text
C:.
│   .gitignore
│   Capture.JPG
│   Dockerfile
│   Readme.md
│   requirements.txt
│
├───data
│       LoansTrainingSetV2.csv
│       Output_Test.csv
│       test_data.csv
│
├───models
│       GBM_Model_version1.pkl
│
├───notebooks
│       loan_eligibility_prediction.ipynb
│       loan_eligibility_test.ipynb
│       SPyder IDE loan eligibilty.py
│       SPyder IDE loan eligibity Test code.py
│
├───results
│       Capture.JPG
│       Capture2.JPG
│
└───src app
    │   app.py
    │
    └───__pycache__
            app.cpython-37.pyc
```
