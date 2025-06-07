# Fraud Detection in Electricity and Gas Consumption - STEG
## Description of the Business Problem
- The Tunisian Company of Electricity and Gas (STEG) is a public and a non-administrative company, it is responsible for delivering electricity and gas across Tunisia. 

- The company suffered tremendous losses in the order of 200 million Tunisian Dinars due to fraudulent manipulations of meters by consumers.

- Using the client’s billing history, we aim to build a model that can detect and recognize clients involved in fraudulent activities.

- The data solution is geared towards enhancing the company’s revenues and reduce the losses caused by such fraudulent activities.

## Data Preparation
1) Exploratory Data Analysis (EDA)

2) Feature Engineering


##### Exploratory Data Analysis
___
**Findings**: 
- There are column names that need renaming to increase ease of understanding: `disrict`, `consommation_level_1`, `reading_remarque` among others
  
- There is a mismatch in the number of regions in the client train (25) and test (24) datasets  - `Region` is an important column thus we need to reconcile the datasets

- There are no missing values in the train sets for both Client and Invoice data

- The Target ('Fraud') is highly imbalanced with *5.6%* of client accounts classified as fraudulent while *94.4%* of client accounts classified as not fraudulent.

  ![Alt Text] (https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/fraud_distribution.png)

- Fraudulent client accounts are reasonably well distributed across districts

   ![Alt Text] (https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/fraud_distribution_district.png)


- There are a few outlier regions with a much higher number of fraudulent client accounts (101-104, 107,311)

  ![Alt Text] (https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/fraud_distribution_region.png)



##### Feature Engineering
___
**Findings:**
- We convert date columns to `date-time` format

- We compute the average consumption levels per client by grouping invoice data by `client_id` then get the mean for each consumption level

- We encode select categorical columns in both test and train datasets: `district`, `client_ctg` and `region`
___

## Data Modelling and Evaluation

#### Methodology
___
- We build 2 baseline models (`Logistic Regression` and `Decision Trees`) to compare both parametric and non-parametric models.

- We then enhance these models with tuned hyperparameters introduce a more robust model (XGBoost)

- We then evaluate these models using ROC AUC and other classification metrics (F1 Score, Accuracy)

#### Summary findings on the Untuned Models
___
**Logistic Regression:**
- The following are the summary metrics for the model:
    - Train and Test Accuracy scores of **~94%**
    - Recall and F1 score of **0-1%** on test data for the minority class (Class 1: Fraud)
    - AUC Score of **~68%** on both test and training data

- These results indicate that the logistic regression model is performing extremely poorly in detecting fraud (Class 1) with a recall score of *zero*. 
- Accuracy scores are extremely misleading (***94%***) which results from us dealing with a highly imbalanced dataset
- The model performs much better than a random model with stable generalization (***AUC: ~68%*** for both train and test data sets indicating *low overfitting*) 

___
**Decision Tree:**
- The following are the summary metrics for the model:
    - Train Accuracy of **100%** and Test Accuracy of **~90%**
    - Recall and F1 score of **13-14%** on test data for the minority class 
    - AUC score of **~100%** on training data (near-perfect model) and **~54%** on test data (slightly better than random) 
- These results indicate classic *overfitting* (*Accuracy/AUC: 100%*) on the training dataset with the model basically memorizing the training data
- The model fails to generalize to the test data and barely performs better than a random model (*ROC-AUC: ~54%*)
- These results also indicate a highly imbalanced dataset.
___
**Logistic vs Decision Tree:**
- The ROC curve indicates that the logistic regression model outperforms the decision tree model in terms of making correct classifications

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/roc_curve_untuned.png)

***Summary statistics***:

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/summ_stats_untuned.png)

___
#### Next Steps: Dealing with Class imbalance + Hyperparameter tuning

- To deal with class imbalance and improve model performance, we will do the following:

    1) Use the `precision-recall curve` to find ***better classification thresholds*** (default=0.5 is rarely optimal on imbalanced data)

    2) Use class weights in our models by adding a new field `class_weight = 'balanced'`

    3) Apply oversampling/undersampling techniques on the minority/majority class (**SMOTE**/**SMOTEEN**)

    4) Apply stratified sampling techniques (**k-fold Cross-validation**)
    
    6) Use a better model to deal with class imbalance: **XGBoost**
 
### Train + Evaluate Tuned Models
___
#### Pipeline 1: Manual Resampling
**Findings:**
- Since our `scale_pos_weight = 0.85` after resampling, this means that our classes are mostly balanced (with the minority class slightly overrepresented)

- In this pipeline, we carry out resampling (using SMOTEENN) once before model training. SMOTE also uses full training data

- This comes with a risk of data leakage and inflated performance on the training set as seen in the table below

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/summ_stats_tuned_pipe1.png)

- As a result of this data leakage, later evaluation (cross validation) is not honest and the Train/Test gap is very wide (poor generalization)

#### Pipeline 2: Using `ImbPipeline`
- To mitigate the issue of inflated model performance due to data leakage, we build our pipeline using `ImbPipeline` which allows us to do the following: 

    a) Apply SMOTEENN within each CV fold

    b) Apply SMOTE only within each training fold - meaning no leakage from the validation set into synthetic data

- Model results from Pipeline 2 thus end up looking like this:

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/summ_stats_tuned_pipe2.png)

- We can see that this pipeline gives a truer picture of model performance on the training dataset - the Train/Test gap is less wide making the model more reliable (great generalization)

#### Summary Findings on the Tuned Models (Pipeline 2)
___
**Logistic Regression (Tuned)**
- The following are the summary metrics for the tuned model:
    - Train/Test Accuracy: **82%**
    - Recall: **37%**, F1 score: **19%** - on test data for the minority class (Class 1: Fraud)
    - AUC Score: **70%** on test data

- The tuned model shows a significant increase in performance on all metrics: recall (increase of ***36%***), F1 score (increase of ***18%***) 

___
**Decision Tree (Tuned)**
- The following are the summary metrics for the tuned model:
    - Train/Test Accuracy: **79%**
    - Recall: **49%**, F1 score: **21%** - on test data for the minority class (Class 1: Fraud)
    - AUC Score: **71%** on test data

- The tuned model shows a significant increase in performance on all metrics when compared to the base models: recall (increase of ***35%***), F1 score (increase of ***7%***) 
___
**XGBoost (Tuned)**
- The following are the summary metrics for the tuned model:
    - Train Accuracy: **86%**, Test Accuracy: **84%**
    - Recall: **37%**, F1 score: **21%** - on test data for the minority class (Class 1: Fraud)
    - AUC Score: **74%** on test data

- As per the ROC curves, `XGBoost` outperforms all other models in terms of predictive power (`AUC = 0.74`)

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/roc_curve_tuned_all.png)

- In terms of the other classification metrics (accuracy, F1-score), `XGBoost` also marginally outperforms other models (1st in accuracy, 1st in F1 score)


***Summary statistics***

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/summ_stats_tuned_pipe2(detailed).png)

#### Summary Findings on the Best Model (Tuned XGBoost)
___
- The following are the summary metrics for XGBoost tuned via GridSearchCV:

    - Train/Test Accuracy: **85%**
    
    - Recall: **39%**, F1 score: **22%** - on test data for the minority class (Class 1: Fraud)
    
    - AUC Score: **74%** on test data
    
    - As per the ROC curve, XGBoost tuned via GridSearchCV also has an `AUC = 0.74`
    
    ![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/roc_curve_tuned_best.png)
    
***Summary Statistics***

![Alt Text](https://github.com/potterblack-ke/Fraud_detection_Project/blob/main/assets/summ_stats_tuned_best.png)

## Conclusion and Next Steps
___
#### Conclusions

- The model with the highest predictive power/performance is:
    - Name: **Tuned XGBoost** 
    
    - Tuning technique: **GridSearchCV**
    
    - Best Model (ROC)AUC Score: **74%**
    
    - Best Model F1 Score: **22%**

#### Next Steps: 

- **Feature transparency**: Identify the top features most predictive of fraud using SHAP: region or district? Consumption threshold? 

- Incorporate Precision-Recall AUC (better than ROC AUC for imbalance)

- Deploy the model onto the client systems

