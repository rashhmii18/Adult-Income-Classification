# Adult Income Classification

## Project Overview

This project focuses on predicting whether a person's annual income is <=50K or >50K using demographic and employment-related information.

The project follows a complete machine learning workflow, starting from data loading and exploratory data analysis to data cleaning, feature engineering, preprocessing, model training, evaluation, model saving, and deployment through a Streamlit prototype.

## Problem Statement

Income classification is a binary classification problem where the goal is to predict whether an individual's annual income is:

* <=50K
* > 50K

The prediction is based on features such as age, education, occupation, workclass, marital status, hours worked per week, capital gain, capital loss, and other demographic information.

## Dataset

The dataset used in this project is the Adult Income Dataset.

Dataset source:

https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

The dataset contains information about individuals and their demographic and employment characteristics.

### Dataset Information

* Original records: 48,842
* Original features: 15
* Target variable: income
* Classification type: Binary classification
* Target classes: <=50K and >50K

## Project Workflow

1. Data Loading
2. Data Inspection
3. Exploratory Data Analysis
4. Data Cleaning
5. Feature Engineering
6. Encoding Categorical Features
7. Feature Scaling
8. Model Training
9. Cross-Validation
10. Model Evaluation
11. Model Saving
12. Streamlit Prototype

## Exploratory Data Analysis

Several EDA techniques were used to understand the dataset and identify relationships between the features and income.

### Income Distribution

Before cleaning, the dataset contained:

* <=50K: 37,155 records
* > 50K: 11,687 records

The target variable is therefore imbalanced, with more observations belonging to the <=50K class.

Because of this imbalance, accuracy was not considered sufficient by itself. Precision, recall, F1-score, and the confusion matrix were also used for evaluation.

### Age

Most individuals in the dataset are between approximately 20 and 50 years old.

The age distribution is slightly right-skewed, with fewer individuals at older ages.

Individuals belonging to the >50K class generally have a higher median age than those in the <=50K class.

### Education

Education shows a noticeable relationship with income.

Individuals with higher educational qualifications, such as Bachelor's, Master's, Professional School, and Doctorate degrees, have a higher proportion of individuals earning >50K.

### Workclass

The Private workclass contains the largest number of records.

Other workclass categories, including self-employed and government-related categories, show different income distributions.

### Occupation

Occupation also shows a noticeable relationship with income.

Occupations such as Exec-managerial and Prof-specialty contain a relatively higher proportion of individuals earning >50K.

### Hours Per Week

Individuals earning >50K generally work more hours per week on average.

However, there is considerable overlap between the two income groups.

### Gender

The income distribution differs between gender categories, with a higher proportion of males in the >50K group.

This is an observed association in the dataset and does not mean that gender itself causes higher income.

### Marital Status

Married-civ-spouse contains a noticeably higher proportion of individuals earning >50K compared with several other marital-status categories.

This relationship may also be associated with other factors such as age, occupation, and education.

### Numerical Features

The correlation analysis showed relatively weak correlations among the numerical features.

The strongest numerical correlation was only around 0.14, indicating that there was no major multicollinearity problem among the numerical variables.

The boxplots also showed that fnlwgt, capital-gain, and capital-loss contain highly skewed values.

## Data Cleaning

The dataset contained several data-quality issues.

### Unknown Values

The dataset used ? to represent unknown values in:

* workclass
* occupation
* native-country

These values were replaced with missing values using NaN.

The missing categorical values were then filled using the mode of their respective columns.

### Duplicate Records

Duplicate records were identified and removed.

A total of 53 duplicate rows were removed.

### Final Dataset

After cleaning:

* Missing values: 0
* Duplicate rows: 0
* Final rows: 48,789
* Final columns: 15

The cleaned dataset was then used for feature engineering and model development.

## Feature Engineering

Two binary features were created from the capital gain and capital loss variables.

### has_capital_gain

This feature indicates whether an individual has any capital gain.

* 1 means the individual has capital gain.
* 0 means the individual has no capital gain.

### has_capital_loss

This feature indicates whether an individual has any capital loss.

* 1 means the individual has capital loss.
* 0 means the individual has no capital loss.

These features were created because the original capital-gain and capital-loss variables contain many zero values.

The fnlwgt feature was also removed because it represents a sampling weight rather than a direct personal characteristic and has a highly skewed distribution.

## Preprocessing

The dataset contains both numerical and categorical features, so different preprocessing techniques were applied.

### Categorical Features

Categorical features were converted into numerical form using One-Hot Encoding.

OneHotEncoder was configured with handle_unknown='ignore' so that the model can safely handle unseen categories during prediction.

### Numerical Features

Numerical features were standardized using StandardScaler.

Scaling was applied to:

* age
* educational-num
* capital-gain
* capital-loss
* hours-per-week
* has_capital_gain
* has_capital_loss

The preprocessing steps were included inside a Scikit-learn Pipeline so that the same preprocessing is applied during both training and prediction.

## Train-Test Split

The cleaned dataset was divided into training and testing sets.

* Training data: 80%
* Testing data: 20%

Stratified splitting was used so that the class distribution of the target variable remained similar in both sets.

The final test set contained 9,758 records.

## Model Training

Several classification models were experimented with during the project, including:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors

Logistic Regression was used as the final model based on the experimental results.

## Final Model

The final model is Logistic Regression.

The model was implemented together with preprocessing using a Scikit-learn Pipeline.

This means that categorical encoding, numerical scaling, and prediction are handled together.

The pipeline also helps prevent data leakage because preprocessing is fitted using the training data.

## Model Validation

5-fold cross-validation was performed on the training data using macro F1-score.

The cross-validation F1-scores were:

* Fold 1: 0.7820
* Fold 2: 0.7846
* Fold 3: 0.7789
* Fold 4: 0.7916
* Fold 5: 0.7917

Mean cross-validation F1-score:

78.58%

This shows that the model produced relatively consistent results across different training and validation splits.

## Model Evaluation

The final model was evaluated on the unseen test dataset.

### Test Results

* Accuracy: approximately 85%
* Macro F1-score: approximately 78%

| Class         | Precision | Recall | F1-score |
| ------------- | --------: | -----: | -------: |
| <=50K         |      0.88 |   0.93 |     0.91 |
| >50K          |      0.74 |   0.60 |     0.66 |
| Macro Average |      0.81 |   0.77 |     0.78 |

The model performs better on the <=50K class than on the >50K class.

This difference is partly related to the imbalance in the dataset, where the <=50K class contains considerably more observations.

### Confusion Matrix

The confusion matrix contained:

* True Negatives: 6,925
* False Positives: 497
* False Negatives: 932
* True Positives: 1,404

The model correctly identifies most <=50K cases but misses some individuals belonging to the >50K class.

Therefore, the model should not be evaluated using accuracy alone.

## Model Saving

The trained model and preprocessing pipeline were saved using Joblib.

The saved model is located at:

model/adult_income_logistic_regression.pkl

The complete pipeline is saved rather than only the Logistic Regression model.

This allows the same preprocessing steps to be automatically applied when making predictions in the prototype.

## Working Prototype

A separate Streamlit prototype was created for the project.

The prototype allows a user to enter information about an individual, such as:

* Age
* Workclass
* Education
* Marital status
* Occupation
* Relationship
* Race
* Gender
* Capital gain
* Capital loss
* Hours per week
* Native country

The application then uses the saved machine learning pipeline to predict whether the individual's income belongs to:

* <=50K
* > 50K

The prototype code is stored separately in app.py.

## Project Structure

Adult-Income-Classification/

```
adult_income.ipynb
app.py
requirements.txt
README.md
model/
    adult_income_logistic_regression.pkl
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## Requirements

The project uses the following Python packages:

streamlit
pandas
joblib
scikit-learn==1.6.1

Scikit-learn version 1.6.1 is used to maintain compatibility with the saved model.

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/rashhmii18/Adult-Income-Classification.git
```

### 2. Open the Project Folder

```
cd Adult-Income-Classification
```

### 3. Create a Virtual Environment

```
python -m venv .venv
```

### 4. Activate the Virtual Environment

On Windows:

```
.venv\Scripts\activate
```

### 5. Install the Requirements

```
pip install -r requirements.txt
```

### 6. Run the Streamlit Prototype

```
streamlit run app.py
```

The application will open in the browser.

## Key Findings

The exploratory analysis showed that several features have noticeable associations with income classification.

Important observations include:

* Higher education levels are generally associated with a higher proportion of >50K income.
* Occupation shows noticeable differences between income groups.
* Older individuals generally have a higher proportion of >50K income.
* Individuals earning >50K generally work more hours per week.
* Marital status shows noticeable differences between the income groups.
* The numerical features do not show strong correlations with each other.
* The dataset is imbalanced, with considerably more <=50K observations.
* Capital gain and capital loss contain many zero values and highly skewed distributions.

## Conclusion

This project demonstrates a complete machine learning workflow for binary income classification.

The Adult Income dataset was first explored and cleaned by handling unknown values and duplicate records. Feature engineering was then performed to create additional binary indicators for capital gain and capital loss.

Categorical variables were converted using One-Hot Encoding, while numerical variables were standardized using StandardScaler. Multiple classification models were considered, and Logistic Regression was used as the final model.

The final model achieved approximately 85% accuracy and a macro F1-score of approximately 78% on the test dataset. Cross-validation also produced a mean macro F1-score of approximately 78.58%, showing reasonably consistent performance.

Finally, the trained pipeline was saved using Joblib and integrated into a separate Streamlit prototype for interactive income prediction.
