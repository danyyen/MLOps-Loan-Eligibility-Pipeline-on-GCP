## MLOps On GCP

This repository contains the code files involved in creating an optimal MLOps Pipeline on GCP (Google Cloud Platform).

### Steps:
* Clone the repository
* Create a Flask App  (app.py)
* Build a Dockerfile

Once the files are created, create a new repository and commit the changes. From here on, this will be your source repository. Proceed with the below steps

###### Cloud Build Trigger
* In your GCP concole, create a new cloud build trigger.
* Point the trigger to your source repository

###### Cloud Run 
* In Cloud Run, point the CI/CD server towards you cloud build trigger out
* The output from cloud build will be in Artifacts Registry which holds a docker image.
* Cloud run will provide a endpoint, a HTTPS URL which will serve the flask app that is created
* Add the permission "allUsers" with roles as "Cloud Run Invoker" and save the changes
* Once changes the change reflects, the HTTPS URL will be accessible
