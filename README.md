
# 🌧️ Rainfall Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Binary%20Classification-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

## 1. Project Overview

**Rainfall Prediction Using Machine Learning** is a supervised machine learning project that predicts whether rainfall is expected based on user-provided weather measurements.

The project uses a **Random Forest Classifier** trained on weather data. It covers the end-to-end machine learning workflow, including data cleaning, missing-value imputation, feature selection, class balancing, model training, hyperparameter tuning, evaluation, model serialization, and web application development.

A Streamlit application allows users to enter seven weather measurements and receive a rainfall or no-rainfall prediction using the saved machine learning model.

This project demonstrates practical skills in Python, data preprocessing, classification, model evaluation, Scikit-learn, and interactive machine learning applications.

## 2. Live Demo

**Live Application:** [Open Rainfall Prediction App](YOUR_STREAMLIT_APP_URL_HERE)

> Replace `YOUR_STREAMLIT_APP_URL_HERE` with the actual public URL of your deployed Streamlit application.

The application is deployed through Streamlit Community Cloud. The live URL is a placeholder until the actual address is added.

## 3. Problem Statement

Rainfall is influenced by several atmospheric and weather-related variables, including pressure, humidity, dew point, cloud cover, sunshine, and wind conditions.

The objective of this project is to use historical weather observations to train a machine learning classifier that identifies whether rainfall is expected for a given set of input measurements.

The project formulates rainfall prediction as a **binary classification problem**, with two possible outcomes:

- **Rainfall:** The model predicts the positive class (`1`).
- **No Rainfall:** The model predicts the negative class (`0`).

The prediction is generated from the trained model and user-provided measurements. It is an educational machine learning demonstration, not an official meteorological forecast.

## 4. Project Objectives

- Clean and prepare weather data for machine learning.
- Handle missing values using suitable statistical imputation methods.
- Encode the target variable into binary numerical values.
- Identify and remove highly correlated temperature-related features.
- Balance the target classes using resampling.
- Select relevant weather measurements as model inputs.
- Train a Random Forest classification model.
- Tune hyperparameters using GridSearchCV and five-fold cross-validation.
- Evaluate model performance using cross-validation and held-out test data.
- Serialize the trained model and feature names using Pickle.
- Develop an interactive Streamlit application for rainfall prediction.
- Deploy the application using Streamlit Community Cloud.

## 5. Key Features

- **Weather-based classification:** Predicts rainfall or no rainfall from seven weather measurements.
- **Data preprocessing:** Handles missing values and prepares the target variable.
- **Feature selection:** Removes selected highly correlated temperature features.
- **Class balancing:** Uses resampling to address the imbalance between rainfall classes.
- **Random Forest model:** Uses an ensemble of decision trees for classification.
- **Hyperparameter optimization:** Searches multiple Random Forest configurations using GridSearchCV.
- **Cross-validation:** Estimates model performance across five validation folds.
- **Model persistence:** Saves the trained estimator and its expected input feature names.
- **Interactive interface:** Accepts weather measurements through Streamlit widgets.
- **Cloud deployment:** Makes the application accessible through a public Streamlit URL once deployed.

## 6. Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, cleaning, transformation, and tabular input preparation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization library |
| Seaborn | Statistical visualization library |
| Scikit-learn | Data splitting, Random Forest, GridSearchCV, cross-validation, and evaluation |
| Random Forest Classifier | Supervised binary classification |
| GridSearchCV | Hyperparameter tuning |
| Cross-validation | Model performance estimation |
| Resampling | Target-class balancing |
| Pickle | Saving and loading the trained model and metadata |
| Streamlit | Interactive web application |
| Jupyter Notebook | Development and experimentation environment |

## 7. Dataset Information

The project was developed using a dataset named `Rainfall.csv`.

The original dataset contains approximately:

| Dataset property | Details |
|---|---|
| File name | `Rainfall.csv` |
| Approximate rows | 366 |
| Original columns | 12 |
| Problem type | Binary classification |
| Target column | `rainfall` |

The dataset contains weather measurements used to predict whether rainfall is expected.

The original `day` column was removed during preprocessing. Three highly correlated temperature-related features were also removed before the final seven input features were selected.

**Dataset availability:** `Rainfall.csv` was used during development. Its presence in the public GitHub repository has not been verified, so this README does not claim that the dataset is included.

If the dataset is not included in the repository, users will need the original file to reproduce the data preparation and model training workflow.

## 8. Machine Learning Workflow

The project follows this general workflow:

1. Import the required Python libraries.
2. Load `Rainfall.csv` using Pandas.
3. Clean the column names and remove the `day` column.
4. Handle missing values in `winddirection` and `windspeed`.
5. Convert the rainfall target into numerical binary values.
6. Remove the selected highly correlated temperature features.
7. Balance the target classes through resampling.
8. Select the seven final input features.
9. Split the data into training and test sets.
10. Train and tune a Random Forest Classifier.
11. Evaluate cross-validation and held-out test performance.
12. Save the best estimator and expected feature names using Pickle.
13. Load the saved model in the Streamlit application.
14. Collect user inputs and display a rainfall prediction.

### Workflow summary

```text
Rainfall.csv
     |
     v
Data Loading and Column Cleaning
     |
     v
Missing-Value Imputation
     |
     v
Target Encoding and Feature Selection
     |
     v
Target-Class Resampling
     |
     v
Train-Test Split (80:20)
     |
     v
Random Forest Classifier
     |
     v
GridSearchCV (5-Fold Cross-Validation)
     |
     v
Model Evaluation
     |
     v
Save Model and Feature Names with Pickle
     |
     v
Streamlit Application
     |
     v
Rainfall / No Rainfall Prediction
```

**Important:** This diagram describes the supplied implementation order. Because resampling occurs before the train-test split in the current workflow, the evaluation may be affected by data leakage. See the limitations section for details.

## 9. Data Preprocessing and Feature Engineering

Data preprocessing prepares the weather measurements for model training and prediction.

### 9.1 Column Cleaning

The dataset column names were cleaned to make them consistent and easier to use in Python.

The `day` column was removed because it was not retained as a model input.

### 9.2 Missing-Value Imputation

Missing values were handled using two different statistical approaches.

| Feature | Imputation method | Reason |
|---|---|---|
| `winddirection` | Mode | Replaces missing values with the most frequently occurring direction value |
| `windspeed` | Median | Uses the middle observed value and is less sensitive to extreme values than the mean |

These operations help prevent missing measurements from interfering with model training.

### 9.3 Target Encoding

The categorical target column `rainfall` was converted into binary numerical values.

| Original value | Encoded value |
|---|---:|
| `yes` | 1 |
| `no` | 0 |

This allows the classifier to work with a numerical target.

### 9.4 Feature Selection

The following features were removed because they were identified as highly correlated temperature-related variables:

- `maxtemp`
- `temparature`
- `mintemp`

Removing highly correlated variables can reduce redundant information. However, correlation alone does not establish that a feature is unhelpful for prediction; feature importance and validation experiments would be useful for confirming the decision.

### 9.5 Final Input Features

The model uses the following seven features:

| Feature | Description |
|---|---|
| `pressure` | Atmospheric pressure |
| `dewpoint` | Dew point temperature |
| `humidity` | Relative humidity |
| `cloud` | Cloud measurement |
| `sunshine` | Sunshine measurement |
| `winddirection` | Wind direction |
| `windspeed` | Wind speed |

The feature names above reflect the supplied implementation. The precise measurement units and valid input ranges should be confirmed from the dataset documentation before interpreting individual predictions.

### 9.6 Class Balancing Through Resampling

Resampling was used to balance the target classes. The resampling operation used `random_state=17`, and the combined data was shuffled using `random_state=42`.

Class balancing can help a classifier learn from a minority class that might otherwise receive insufficient attention during training.

However, resampling has limitations:

- It can change the class distribution seen by the model.
- It does not guarantee better predictions on naturally distributed data.
- It can increase the risk of overfitting when observations are duplicated.
- If performed before splitting the dataset, it can introduce information leakage between training and test data.

A more reliable approach is to split the original data first and apply resampling only to the training data. For cross-validation, resampling should occur separately within each training fold, preferably through an appropriate Scikit-learn-compatible pipeline.

## 10. Model Training and Hyperparameter Tuning

### 10.1 Random Forest Classifier

The project uses `RandomForestClassifier`, an ensemble learning algorithm that combines predictions from multiple decision trees.

For classification, the trees collectively determine the predicted class. Random Forest can model nonlinear relationships and interactions among weather features without requiring all relationships to be specified manually.

### 10.2 Train-Test Split

The dataset was divided using `train_test_split` with the following settings:

| Parameter | Value |
|---|---|
| Test size | 0.20 (20%) |
| Training proportion | 80% |
| Random state | 17 |

The held-out test set is intended to provide an evaluation separate from the cross-validation process used during model selection.

However, because the current workflow performs resampling before the split, this test set may not be fully independent of the original observations used to create the resampled dataset.

### 10.3 Hyperparameter Tuning with GridSearchCV

`GridSearchCV` was used to search for a suitable Random Forest configuration through five-fold cross-validation.

The search included the following parameter combinations:

| Hyperparameter | Candidate values |
|---|---|
| `n_estimators` | 50, 100, 150, 200 |
| `max_features` | `sqrt`, `log2` |
| `max_depth` | `None`, 5, 10, 20, 30 |
| `min_samples_split` | 2, 5, 7, 10 |
| `min_samples_leaf` | 1, 2, 4 |

Additional GridSearchCV settings:

| Setting | Value |
|---|---|
| Cross-validation folds | 5 |
| `n_jobs` | -1 |
| `verbose` | 2 |
| Best estimator variable | `best_rf_model` |

`n_jobs=-1` allows Scikit-learn to use the available CPU cores for parallel computation where supported. `verbose=2` displays progress information while the grid search runs.

The selected estimator is stored in `best_rf_model` for evaluation and serialization.

## 11. Model Evaluation Results

The recorded model evaluation results are:

| Evaluation metric | Result |
|---|---:|
| Mean cross-validation score | 0.8024 (approximately 80.24%) |
| Held-out test accuracy | 0.7021 (approximately 70.21%) |

### Understanding the results

**Mean cross-validation score — approximately 80.24%**

This is the recorded mean score across the cross-validation folds used during model selection. It summarizes performance over multiple validation splits; it is not the final test accuracy.

**Held-out test accuracy — approximately 70.21%**

This is the recorded accuracy on the test split. Based on the confusion matrix, the classifier correctly classified 33 of 47 test observations, resulting in approximately 70.21% accuracy.

The difference between the cross-validation score and test accuracy suggests that performance varies between evaluation procedures. Possible explanations include sampling variability, model-selection effects, class balancing, and potential data leakage from resampling before the split.

These results should be treated as preliminary rather than proof of reliable real-world rainfall prediction. A fresh evaluation using an untouched, naturally distributed test set is recommended.

## 12. Confusion Matrix Explanation

The recorded confusion matrix is:

```text
[[14, 10],
 [ 4, 19]]
```

Using the standard Scikit-learn binary classification label order `[0, 1]`, the rows represent the actual classes and the columns represent the predicted classes.

| Actual class | Predicted No Rainfall (0) | Predicted Rainfall (1) |
|---|---:|---:|
| No Rainfall (0) | 14 | 10 |
| Rainfall (1) | 4 | 19 |

### Interpretation

- **True Negatives (TN): 14** — Actual no-rainfall observations correctly predicted as no rainfall.
- **False Positives (FP): 10** — Actual no-rainfall observations incorrectly predicted as rainfall.
- **False Negatives (FN): 4** — Actual rainfall observations incorrectly predicted as no rainfall.
- **True Positives (TP): 19** — Actual rainfall observations correctly predicted as rainfall.

The matrix contains 47 test observations.

Accuracy is calculated as:

`Accuracy = (TN + TP) / Total Observations`

`Accuracy = (14 + 19) / 47 ≈ 70.21%`

The confusion matrix also shows why accuracy alone is insufficient: false rainfall alerts and missed rainfall events have different practical implications. Precision, recall, F1-score, specificity, and balanced accuracy should be assessed in future evaluations.

## 13. Streamlit Application

The project includes an interactive Streamlit web application in `app.py`.

The application loads the serialized Random Forest model and collects the required weather measurements from the user.

### Input fields

Users can enter the following measurements:

| Input field | Model feature |
|---|---|
| Pressure | `pressure` |
| Dew Point | `dewpoint` |
| Humidity | `humidity` |
| Cloud | `cloud` |
| Sunshine | `sunshine` |
| Wind Direction | `winddirection` |
| Wind Speed | `windspeed` |

### Prediction process

1. The application loads `rainfall_prediction_model.pkl`.
2. It retrieves the trained model and saved feature names.
3. The user enters the seven weather measurements.
4. The application creates a Pandas DataFrame containing the inputs.
5. The DataFrame columns are arranged to match the stored feature order.
6. When the user clicks **Predict Rainfall**, the model generates a classification.
7. The application displays either **Rainfall** or **No Rainfall**.

The interface also displays a disclaimer that the result is an educational machine learning prediction rather than an official weather forecast.

### Why feature order matters

A trained Scikit-learn model expects input features in the same order used during training.

For example, if the model expects `pressure` first and `dewpoint` second, swapping those values can cause the model to interpret one measurement as another. This may produce incorrect predictions even if every input is numerical.

Storing and reusing `feature_names` helps the application arrange input data consistently with the trained model.

## 14. Project Folder Structure

The following is the **expected structure for the deployed application**, based on the project information supplied. It is not a verified listing of the current GitHub repository.

```text
rainfall-prediction/
├── app.py
├── rainfall_prediction_model.pkl
├── requirements.txt
└── README.md
```

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface and prediction logic |
| `rainfall_prediction_model.pkl` | Serialized model and feature-name metadata |
| `requirements.txt` | Python dependency list |
| `README.md` | Project documentation |

`Rainfall.csv` and the development notebook are not listed in this deployment structure because their presence in the repository has not been verified.

## 15. Installation and Setup

Follow these instructions on Windows to run the application locally.

### Prerequisites

Install the following:

- Python compatible with the project's dependencies.
- Git.
- A terminal such as Command Prompt, PowerShell, or the VS Code terminal.

### Step 1: Clone the repository

Open a terminal and execute:

```bash
git clone https://github.com/SelvaSankar17/rainfall-prediction.git
cd rainfall-prediction
```

### Step 2: Create a virtual environment

Creating a separate environment helps avoid conflicts with packages installed for other Python projects.

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```bat
.venv\Scripts\activate
```

Or activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of its execution policy, use Command Prompt or follow your organization's approved Python environment setup.

### Step 3: Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project's dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include the packages required by `app.py`, including Streamlit, Pandas, and the appropriate Scikit-learn dependency.

No specific package versions are asserted here because the actual dependency file has not been inspected.

**Important model compatibility note:** Scikit-learn does not guarantee that Pickle-serialized estimators will load reliably across different library versions. For the most reproducible setup, use the compatible Scikit-learn version from the original training environment and pin tested dependency versions in `requirements.txt`.

## 16. How to Run the Application Locally

After cloning the repository, activating the environment, and installing the dependencies, run:

```bash
streamlit run app.py
```

Streamlit normally opens the application in your default browser at:

[http://localhost:8501](http://localhost:8501)

If the browser does not open automatically, copy that address into your browser.

Keep the terminal running while using the application. To stop the local server, press `Ctrl+C` in the terminal.

If the application fails to start, check that:

- `app.py` exists in the current directory.
- `requirements.txt` has been installed successfully.
- `rainfall_prediction_model.pkl` exists at the path expected by the application.
- The saved model can be loaded using the installed Python and Scikit-learn versions.
- The feature names and input data match the trained model's expected inputs.

## 17. How to Use the Application

1. Open the local application or the public Streamlit URL.
2. Enter the pressure measurement.
3. Enter the dew point measurement.
4. Enter the humidity measurement.
5. Enter the cloud measurement.
6. Enter the sunshine measurement.
7. Enter the wind direction.
8. Enter the wind speed.
9. Click **Predict Rainfall**.
10. Read the displayed classification: **Rainfall** or **No Rainfall**.

Use values within the valid ranges and units of the original dataset. Since the input units and complete validation ranges have not been specified here, they should be documented in the application before the tool is used for any practical weather-related decision.

## 18. Streamlit Community Cloud Deployment

The application has been deployed through Streamlit Community Cloud. The following steps describe the deployment workflow for setting up or updating the hosted application.

### Deployment steps

1. Push the application files to your GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Sign in using your GitHub account.
4. Choose the option to create or deploy an application.
5. Select the repository `SelvaSankar17/rainfall-prediction`.
6. Select the branch containing the application files.
7. Set the main file path to `app.py`.
8. Configure the Python and dependency environment as required.
9. Deploy the application.
10. Wait for the deployment to complete and check the application logs if any errors occur.
11. Open the public application URL and test the prediction flow.
12. Replace the Live Demo placeholder in this README with the actual public URL.

### Files required for deployment

The application deployment should include the files needed to start the Streamlit interface and load the trained model:

- `app.py`
- `rainfall_prediction_model.pkl`
- `requirements.txt`

The README should also be committed to the repository.

If the dataset is only required during model development and is not loaded by `app.py`, it does not necessarily need to be part of the deployed application. Confirm this against the actual application code before removing or excluding any files.

### Accessing the public application

Once deployed successfully, users can open the public Streamlit URL in a browser and make predictions without installing Python or running the application locally.

The public app URL must be added to the Live Demo section above after it has been verified.

## 19. Challenges and Limitations

### 19.1 Limited Dataset Size

The original dataset contains approximately 366 rows and 12 columns. This is a relatively small dataset for modelling weather-related patterns, and its observations may not represent all locations, seasons, or weather conditions.

A larger and more diverse dataset would help assess whether the model generalizes beyond the development data.

### 19.2 Difference Between Cross-Validation and Test Results

The mean cross-validation score is approximately 80.24%, while the recorded test accuracy is approximately 70.21%.

This gap indicates that the evaluation results are not identical across the two procedures. More rigorous validation is required before drawing conclusions about expected performance on new weather observations.

### 19.3 Potential Data Leakage from Resampling

Resampling was performed before the train-test split in the supplied workflow.

If resampling duplicates observations or otherwise allows related information to occur in both training and test sets, the test evaluation may be overly optimistic because the two sets are no longer fully independent.

A recommended improvement is to split the original dataset first, keep the test set untouched, and apply resampling only to training data. During cross-validation, resampling should be performed separately within each training fold.

### 19.4 Class Imbalance and Real-World Distribution

Balancing the training classes changes their proportions relative to the original data.

As a result, a model trained on resampled data may behave differently when it encounters naturally occurring rainfall frequencies. Precision, recall, calibration, and performance on an untouched dataset should be examined.

### 19.5 Feature Selection

Three temperature-related features were removed due to high correlation.

Although reducing redundant features can be useful, correlated variables may still carry predictive information. Feature selection should be validated using training-only experiments rather than correlation alone.

### 19.6 Limited Evaluation Metrics

The recorded evaluation includes cross-validation score, test accuracy, and a confusion matrix.

Further evaluation using precision, recall, F1-score, specificity, balanced accuracy, and appropriate probability-based metrics would provide a more complete understanding of the model's performance.

### 19.7 No Official Weather Data Integration

The Streamlit application uses manually entered weather measurements and a saved model. It is not connected to a live meteorological data feed.

Its output should not be interpreted as an official forecast or as a prediction of rainfall at a specific future time unless the model is redesigned and validated for that purpose.

### 19.8 Pickle Security and Compatibility

Pickle files can execute arbitrary code when deserialized. Only load model files from trusted sources.

The model file must also remain compatible with the Python and Scikit-learn environment in which it is loaded.

### 19.9 Generalization and Reliability

The recorded results alone do not establish that the model is reliable for operational weather forecasting. Independent validation on a larger dataset, with suitable temporal or location-based splits where appropriate, would be necessary before considering real-world use.

## 20. Future Improvements

The following improvements could strengthen the project:

- **Leakage-safe preprocessing:** Split the original dataset before resampling and use a suitable pipeline to apply resampling within each training fold.
- **More representative data:** Collect additional weather observations covering different locations, seasons, and rainfall conditions.
- **Stronger evaluation:** Report precision, recall, F1-score, specificity, balanced accuracy, and a detailed classification report.
- **Independent testing:** Evaluate the final model on untouched data with a naturally occurring class distribution.
- **Feature-selection validation:** Compare alternative feature sets using training-only cross-validation.
- **Model comparison:** Compare Random Forest with other appropriate classification algorithms using the same leakage-safe evaluation process.
- **Input validation:** Define valid ranges and units for every Streamlit input and reject invalid values.
- **Dependency reproducibility:** Record and pin compatible dependency versions, particularly the Scikit-learn version used to serialize and load the model.
- **Improved prediction communication:** Display clear explanations of the predicted class and communicate uncertainty without implying unsupported confidence.
- **Better documentation:** Add verified notebook links, data provenance, and screenshots only when those assets are available in the repository.
- **Weather-specific validation:** If the intended goal is future rainfall forecasting, redesign the target and validation procedure to reflect a defined forecast horizon and test on later time periods.

## 21. Disclaimer

This project is intended for educational purposes and demonstrates a machine learning workflow for binary rainfall classification.

The displayed prediction is generated by a trained model using user-provided measurements. It is not an official weather forecast, and the model has not been established as suitable for operational meteorology or safety-critical decisions.

Do not use its predictions as a substitute for official weather alerts, meteorological services, or emergency guidance.

## 22. Author Information

**SelvaSankar M**

Data Science & AI | SQL | Python | Excel | Power BI

Interested in data science, machine learning, data analysis, and building practical applications that turn data into useful insights.

## 23. GitHub Repository Link

Explore the source code and project documentation:

**[Rainfall Prediction Using Machine Learning — GitHub Repository](https://github.com/SelvaSankar17/rainfall-prediction)**

If you find the project useful, consider starring the repository and sharing feedback or suggestions for improvement.
