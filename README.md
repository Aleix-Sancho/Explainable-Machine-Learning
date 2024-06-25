# Explainable Machine Learning and Credit Risk Models

> Any sufficiently advanced technology is  
> indistinguishable from magic.  
> Arthur C. Clarke  



Until the trick gets revealed for most of us magic is related with the unknown and the unpredictable. Thus, it produces an instinctive reaction of precaution, alertness and curiosity. As an universal law, new advancements in science neither escape this initial fate and Explainable Machine Learning is just another proof for that fact.

This repository contains the code and resources associated with the Bachelor's Thesis titled "Explainable Machine Learning and Credit Risk Models," completed by Aleix Sancho Jiménez and supervised by Dr./Prof. Alejandra Cabaña Nigro.

## Description

The main objective of this project is to address explainability in Machine Learning models applied to credit risk assessment. The study focuses on how cooperative game theory can offer effective techniques to improve the transparency of complex predictive models, such as neural networks.

## Repository Contents

This repository is organized as follows:


- `SHAP_Credit_Risk_Model.py`: Implementation of the credit risk model using SHAP values.
- `credit_risk_dataset.csv`: Dataset used for training and evaluating the model.
- `Lime_Tangent_Plane.py`: Implementation of LIME.
- `SHAP_Framework_Implementations.py`: Implementation of the SHAP framework based on Shapley values.
- `Graphs_Features_Credit_Risk_Model.py`: Graphs and feature analysis of the credit risk model.
- `Techniques_Comparative.py`: Comparative code for different explainability techniques.
- `Approximation_f_S_by_E(f).py`: Approximations and explanations using SHAP values.
- `Explainable_Machine_Learning_and_Credit_Risk_Models.pdf`: Bachelor's thesis document detailing the project and its findings.

## Explainable Techniques

The project covers the following explainable techniques:

- **LIME (Local Interpretable Model-Agnostic Explanations)**: A model-agnostic technique that seeks to explain local predictions of black-box models.
- **DeepLIFT (Deep Learning Important FeaTures)**: A method specific to neural networks that assigns contribution scores to each feature.
- **SHAP Values (SHapley Additive exPlanations)**: Implementation of Shapley values that unify LIME and DeepLIFT techniques to provide a robust, mathematically sound solution to explainability.

## Application to Credit Risk Models

The SHAP framework has been applied to a credit risk model using a specific dataset. The trained model is a dense neural network that predicts loan status (default/no default) based on various financial characteristics of borrowers.
