# ATgfe (Automated Transparent Genetic Feature Engineering)

<div style="text-align:center">
<img src="https://live.staticflickr.com/65535/49014776017_40a14d33ef.jpg" alt="ATgfe-logo"/>
</div>

# What is ATgfe?
ATgfe stands for Automated Transparent Feature Engineering. ATgfe uses genetic algorithm to engineer new features by trying different interactions between 
the existing features and measuring the effectiveness/prediction power of these new features using the selected evaluation metric.

ATgfe applies the following operations to generate candidate features:
- Features interactions using the basic operators (+, -, x, /).
``` 
    (petalwidth * petallength) 
```
- Applying feature transformation with features interactions, these operations can be easily extended using user defined functions.
```
    squared(sepalwidth)*(log_10(sepalwidth)/squared(petalwidth))-cube(sepalwidth)
```
- Adding weights to the features interactions
```
    (0.09*exp(petallength)+0.7*sepallength/0.12*exp(petalwidth))+0.9*squared(sepalwidth)
```
- Use categorical features to create more complex features using groupBy
```
    (0.56*groupByYear0TakeMeanOfFeelslike*0.51*feelslike)+(0.45*temp)
```

# Why ATgfe?
ATgfe allows you to solve linear and **non-linear** problems by generating new **explainable** features from existing features. The generated 
features can then be used with a linear model, you should be able to see substantial improvement if there is extra predictive information that can be gained from
interacting existing features together.

ATgfe has been compared with non-linear models like Gradient Boosting and it seems to achieve comparable results and over-perform them in some cases.
Check the following examples [BMI](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/generated/generated_1.ipynb), [Rational difference](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/generated/generated_2.ipynb) and
[IRIS](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/toy-examples/iris_multi_classification.ipynb)

# Results
## Generated
| Expression                       | Linear Regression                                               | LightGBM Regressor                                             | Linear Regression + ATgfe                               |
|----------------------------------|-----------------------------------------------------------------|----------------------------------------------------------------|---------------------------------------------------------|
| BMI = weight/height^2            | <ul>     <li>RMSE: 3.268</li>     <li>r^2: 0.934</li> </ul>     | <ul>     <li>RMSE: 0.624</li>     <li>r^2: 0.996</li> </ul>    | <ul>  <li>RMSE: **0.0**</li><li>r^2: **1.0**</li> </ul> |
| Y = (X1 - X2) / (X3 - X4)        | <ul>     <li>RMSE: 141.261</li>     <li>r^2: -0.068</li> </ul>  |  <ul>     <li>RMSE: 150.642</li>     <li>r^2: -0.52</li> </ul> | <ul>  <li>RMSE: **0.0**</li><li>r^2: **1.0**</li> </ul> |
| Y = (Log10(X1) + Log10(X2)) / X5 | <ul>     <li>RMSE: 0.140</li>     <li>r^2: 0.899</li> </ul>     | <ul>     <li>RMSE: 0.102</li>     <li>r^2: 0.895</li> </ul>    | <ul>  <li>RMSE: **0.0**</li><li>r^2: **1.0**</li> </ul> |
| Y = 0.4*X2^2 + 2*X4 + 2          | <ul>     <li>RMSE: 30077.269</li>     <li>r^2: 0.943</li> </ul> | <ul>     <li>RMSE: 980.297</li>     <li>r^2: 1.0</li> </ul>    | <ul>  <li>RMSE: **0.0**</li><li>r^2: **1.0**</li> </ul> |

## Classification
| Dataset                          | Logistic Regression                                                     | LightGBM Classifier                                                    | Logistic Regression + ATgfe                                        |
|----------------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| SPAM  (57 features)              | <ul>     <li>Accuracy: 0.917</li>     <li>ROC_AUC: 0.97</li> </ul>      | <ul><li>Accuracy: **0.944**</li>  <li>ROC_AUC: **0.98**</li> </ul>     | <ul>     <li>Accuracy: 0.931</li>     <li>ROC_AUC: 0.97</li> </ul> |
| IRIS  (4 features)               | <ul>     <li>Accuracy: 0.9</li>     <li>ROC_AUC: 0.95</li> </ul>        |  <ul>     <li>Accuracy: 0.946</li>     <li>ROC_AUC: 0.98</li> </ul>    | <ul>  <li>Accuracy: **0.973**</li><li>ROC_AUC: **0.99**</li> </ul> |

## Regression
| Dataset                          | Linear Regression                                                       | LightGBM Regressor                                                     | Linear Regression + ATgfe                                          |
|----------------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| Concrete  (8 features)           | <ul>     <li>RMSE: 11.13</li>     <li>r^2: 0.643</li> </ul>             | <ul>     <li>RMSE: **6.44**</li>     <li>r^2: **0.935**</li> </ul>     | <ul>     <li>RMSE: 6.68</li>     <li>r^2: 0.891</li> </ul>         |
| Boston  (13 features)            | <ul>     <li>RMSE: 4.796</li>     <li>r^2: 0.765</li> </ul>             | <ul>     <li>RMSE: 3.38</li>     <li>r^2: 0.859</li> </ul>             | <ul>  <li>RMSE: **3.20**</li><li>r^2: **0.895**</li> </ul>         |

# Get started

## Requirements
- Python ^3.6
- DEAP ^1.3
- Pandas ^0.25.2
- Scipy ^1.3
- Numpy ^1.17
- Sympy ^1.4

## Install ATgfe
```
pip install atgfe
```
## Upgrade ATgfe
```
pip install -U atgfe
```
# Usage

## Examples
The [Examples](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/) are divided into two sections:
- [Generated](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/generated), where we test ATgfe's ability to handle hand-crafted non-linear problems where we know there is
information that can be produced from interacting features together. 

- [Toy Examples](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/toy-examples)'s datasets which includes a mix of regression
 and classification problems including a multi-label regression problem.

## Clean column names
### ATgfe can't handle special characters
```
# example
def prepare_column_names(columns):
    return [col.replace(';', 'semi')
            .replace('(', 'left_brace')
            .replace('[', 'left_bracket')
            .replace('$', 'dollar_sign')
            .replace('#', 'hash')
            .replace('!', 'explanation_mark')
            for col in columns]

columns = prepare_column_names(df.columns.tolist())
df.columns = columns
```

## GeneticFeatureEngineer arguments
```
GeneticFeatureEngineer(
    model,
    x_train: pandas.core.frame.DataFrame,
    y_train: pandas.core.frame.DataFrame,
    numerical_features: List[str],
    number_of_candidate_features: int,
    number_of_interacting_features: int,
    evaluation_metric: Callable[..., Any],
    minimize_metric: bool = True,
    categorical_features: List[str] = None,
    enable_grouping: bool = False,
    sampling_size: int = None,
    cv: int = 10,
    fit_wo_original_columns: bool = False,
    enable_feature_transformation_operations: bool = False,
    enable_weights: bool = False,
    weights_number_of_decimal_places: int = 2,
    verbose: bool = True,
)
```

### model
You can pass any model or pipeline that follow's scikit-learn's API. It should implement the ```fit()``` and ```predict()``` methods.

### x_train
Pass your training features as a pandas Dataframe.

### y_train
Pass your training labels as a pandas Dataframe to handle multi-labels problems

### numerical_features
Pass your numerical features that you would like ATgfe to consider in the feature engineering optimization. 

### number_of_candidate_features
The maximum number of features that you would like ATgfe to generate

### number_of_interacting_features
The maximum number of features that you would like ATgfe to try different interactions with. 
These features will be selected from the numerical features passed.

### evaluation_metric
Pass the evaluation metric that you would like ATgfe to use. You can pass one of [scitkit-learn metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) or create you own.
```python
import numpy as np
from sklearn.metrics import  mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```
### minimize_metric
Set minimize_metric to True if you would like ATgfe to minimize your evaluation metric
 and set to False if you would like it to maximize the metric.
 
### categorical_features
Pass your categorical_features which you would like to use in the grouping. You need to enable_grouping in order 
for the categorical_features to be utilized.

### enable_grouping
Set to True if you would like to use your categorical features to create more complex features using groupBy.

### sampling_size
Set the sampling_size if you would like to run the optimization with a sample of your data. If the sampling_size
 is greater than the number of observations then ATgfe will sample with replacement.
 
### cv
Every generation, ATgfe evaluates the current best solution using k-fold cross validation.
The default number of folds is 10.

### fit_wo_original_columns
If set to True, ATgfe will fit the model without the original numerical features passed. 
It will only use the engineered features and any remaining features in x_train.

### enable_feature_transformation_operations
If set to True, ATgfe will apply different transformation operations on your numerical_features.
These are the existing operations:
```
np_log(), np_log_10(), np_exp(), squared(), cube()
```
You can easily remove from or add to the existing operations. Check out the next section for examples.

### enable_weights
If set to True, ATgfe will add weights in the engineered features which should help in improving your metric

### weights_number_of_decimal_places
Select the number of decimal places to be used in the weights

### verbose
Set to True to enable the logging.

## fit() function arguments
```
gfe.fit(
    number_of_generations: int = 100,
    population_size: int = 300,
    crossover_probability: float = 0.5,
    mutation_probability: float = 0.2,
    early_stopping_patience: int = 5,
    random_state: int = 77,
)
```

### number_of_generations
Maximum number of generations to go through

### population_size
Number of solutions in a population

### crossover_probability
The crossover probability

### mutation_probability
The mutation probability

### early_stopping_patience
The maximum number of generations to wait before early stopping when the validation score is not improving.


## transform() function arguments
```
X = gfe.transform(X)
```

Where X is the pandas dataframe that you would like to add the generated features to.

## Transformation operations

### Get current transformation operations
```
gfe.get_enabled_transformation_operations()
```

The current enabled transformation operations will be returned.

```
['None', 'np_log', 'np_log_10', 'np_exp', 'squared', 'cube']
```
### Remove existing transformation operations
```gfe.remove_transformation_operation``` accepts string or list of strings
```
gfe.remove_transformation_operation('squared')
```

```
gfe.remove_transformation_operation(['np_log_10', 'np_exp'])
```
### Add new transformation operations 
```
np_sqrt = np.sqrt

def some_func(x):
    return (x * 2)/3

gfe.add_transformation_operation('sqrt', np_sqrt)
gfe.add_transformation_operation('some_func', some_func)
```