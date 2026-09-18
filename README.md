# Spotter ML Freight Rate Assessment

Machine learning project for predicting freight load posted rates using data preprocessing, exploratory data analysis, feature engineering, model comparison, and regression modeling.

## Project Overview

The objective of this project is to predict the `posted_rate` of freight loads using shipment, location, equipment, distance, weight, market, quote, and date-related features.

The project includes:

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Model comparison
- Holdout validation
- Hyperparameter tuning
- Final model training
- Validation predictions
- December 2025 predictions
- Assessment scoring

## Dataset

The development dataset contains 48,000 labeled loads from `train_test.csv`.

The final assessment dataset contains 12,000 loads from `validation.csv`.

### Features

- `id` - Load record identifier
- `load_id` - Unique load identifier
- `pickup` - Pickup location
- `delivery` - Delivery location
- `pickup_lat` - Pickup latitude
- `pickup_lon` - Pickup longitude
- `delivery_lat` - Delivery latitude
- `delivery_lon` - Delivery longitude
- `distance` - Shipment distance
- `equipment` - Equipment type
- `weight` - Load weight
- `date` - Load date
- `market_index` - Market-related index
- `quote_signal` - Quote-related signal
- `posted_rate` - Target variable

## Data Cleaning

### Missing Values

Missing values were identified in `weight` and `market_index`.

Missing values were filled using the median calculated from the development data.

### Weight Values

Some records contained negative weight values.

The negative values were converted to absolute values because their corresponding positive values were within the expected weight range.

### Duplicate Records

No duplicate records were identified in the development dataset.

### Outliers

High `posted_rate` values were identified using the IQR method.

These observations were retained because they represented potentially valid freight-rate observations, particularly for longer-distance loads.

## Exploratory Data Analysis

The relationship between the input features and `posted_rate` was analyzed.

`distance` showed the strongest linear relationship with the target, with a correlation of approximately 0.91.

Other important features investigated included:

- Distance
- Weight
- Equipment
- Pickup and delivery locations
- Market index
- Quote signal
- Date-related features

## Feature Engineering

Several additional features were created.

### Date Features

The date column was used to derive:

- `year`
- `month`
- `day`
- `day_of_week`
- `week_of_year`

### Log-Transformed Distance

A logarithmic transformation was applied to distance:

`distance_log = np.log1p(distance)`

### Weight-Distance Ratio

A weight-distance feature was created:

`weight_distance_ratio = weight / distance`

### Geographic Difference Features

Differences between pickup and delivery coordinates were also investigated:

`lat_diff = delivery_lat - pickup_lat`

`lon_diff = delivery_lon - pickup_lon`

A Haversine-based geographic distance was also tested, but it was highly correlated with the existing `distance` feature and was not included in the final model.

## Validation Approach

The labeled development dataset was divided into 80% training data and 20% holdout test data using `train_test_split` with `random_state=42`.

The development data was not split by month.

The holdout dataset was used to compare model performance.

The separate `validation.csv` dataset was reserved for generating the final assessment predictions.

### Evaluation Metrics

- MAE - Mean Absolute Error
- RMSE - Root Mean Squared Error
- R² - Coefficient of Determination

## Model Comparison

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Linear Regression | 131.19 | 533.85 | 0.8667 |
| Decision Tree | 172.37 | 886.04 | 0.6329 |
| Tuned Decision Tree | 151.99 | 579.76 | 0.8428 |
| Random Forest | 127.79 | 563.65 | 0.8514 |
| HistGradientBoosting | 120.73 | 535.69 | 0.8658 |
| Tuned HistGradientBoosting | 119.54 | 533.44 | 0.8669 |

## Final Model

The selected model was `HistGradientBoostingRegressor`.

Model parameters:

- `max_iter = 200`
- `learning_rate = 0.03`
- `max_leaf_nodes = 31`
- `min_samples_leaf = 20`
- `l2_regularization = 1`
- `random_state = 42`

The final model was trained on the complete 48,000-row development dataset before generating validation predictions.

## City Feature Experiment

Pickup and delivery locations were also tested using one-hot encoding.

The city-based HistGradientBoosting model achieved:

- MAE: 124.35
- RMSE: 534.30
- R²: 0.8665

A Random Forest experiment with city features showed a larger train-test performance gap, indicating stronger overfitting.

## Validation Predictions

The final model was used to generate predictions for all 12,000 loads in `validation.csv`.

The predictions were saved as:

`validation_predictions.csv`

The file contains:

- `load_id`
- `predicted_rate`

## December 2025 Predictions

The assessment also requires predictions for `december_chart_inputs.csv`.

The December inputs contain fixed shipment characteristics with dates across December 2025.

The December predictions were generated using the city-feature model because the December input file does not contain the coordinate, market-index, and quote-signal features required by the main model.

The predictions were written to the `predicted_rate` column while preserving the original input columns.

## Project Workflow

Data Loading
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Train/Test Split
↓
Model Training
↓
Model Comparison
↓
Hyperparameter Tuning
↓
Final Model Training
↓
Validation Predictions
↓
December Predictions
↓
Assessment Scoring

## Repository Structure

spotter-ml-freight-rate-assessment/
│
├── README.md
├── assessment_report.pdf
├── freight_rate_prediction.ipynb
├── validation_predictions.csv
├── december_chart_inputs.csv
├── score.py
│
└── data/
    ├── train_test.csv
    ├── validation.csv
    └── december_chart_inputs.csv

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

### Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- HistGradientBoosting Regressor

## Assessment Outputs

### Validation Predictions

`validation_predictions.csv`

Contains predicted freight rates for all validation loads.

### December Predictions

`december_chart_inputs.csv`

Contains the December 2025 predicted rates required for the assessment chart.

### Assessment Report

`assessment_report.pdf`

Contains the project methodology, preprocessing, validation approach, model comparison, results, and December prediction analysis.

## How to Run

Install the required libraries:

`pip install pandas numpy matplotlib scikit-learn jupyter`

Open the notebook:

`jupyter notebook`

Run the project notebook:

`freight_rate_prediction.ipynb`

Run the assessment scorer:

`python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv`

The scorer validates the prediction files and generates the December prediction chart.

## Final Development Results

The selected tuned HistGradientBoosting model achieved the following results on the 20% development holdout:

- MAE: 119.54
- RMSE: 533.44
- R²: 0.8669

The final model was then trained on all available labeled development data and used to generate predictions for the assessment validation dataset.

## Author

Hari Narayanan.R

B.Tech Artificial Intelligence and Data Science Student
