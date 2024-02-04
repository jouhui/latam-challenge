# Flight delay estimation

## Model

Between both models with data balancing and feature selection, I chose XGBoost
because it can handle possibly non-linear relationships, which is not possible for Linear Regression.

For simplicity, the model file is locally saved in the container, but in a real-world scenario, I would save it in a cloud storage service with model versioning tools.

## API

The API is deployed on GCP using Cloud Run. On each push to the main branch, the Gloud Build workflow is
triggered and the API is deployed.
