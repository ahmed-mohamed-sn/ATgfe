# ATgfe (Automated Transparent Genetic Feature Engineering)

<div style="text-align:center">
<img src="https://live.staticflickr.com/65535/49014776017_40a14d33ef.jpg" alt="ATgfe-logo"/>
</div>

# What is ATgfe?
ATgfe stands for Automated Transparent Genetic Feature Engineering. ATgfe is powered by genetic algorithm to engineer new features. The idea is to compose new interpretable features based on interactions between the existing features. The predictive power of the newly constructed features are measured using a pre-defined evaluation metric, which can be custom designed.

ATgfe applies the following techniques to generate candidate features:
- Simple feature interactions by using the basic operators (+, -, *, /).
``` 
    (petalwidth * petallength) 
```
- Scientific feature interactions by applying transformation operators (e.g. log, cosine, cube, etc. as well as custom operators which can be easily implemented using user defined functions).
```
    squared(sepalwidth)*(log_10(sepalwidth)/squared(petalwidth))-cube(sepalwidth)
```
- Weighted feature interactions by adding weights to the simple and/or scientific feature interactions.
```
    (0.09*exp(petallength)+0.7*sepallength/0.12*exp(petalwidth))+0.9*squared(sepalwidth)
```
- Complex feature interactions by applying groupBy on the categorical features.
```
    (0.56*groupByYear0TakeMeanOfFeelslike*0.51*feelslike)+(0.45*temp)
```

# Why ATgfe?
ATgfe allows you to deal with **non-linear** problems by generating new **interpretable** features from existing features. The generated features can then be used with a linear model, which is inherently explainable. The idea is to explore potential predictive information that can be represented using interactions between existing features.

When compared with non-linear models (e.g. gradient boosting machines, random forests, etc.), ATgfe can achieve comparable results and in some cases over-perform them.
This is demonstrated in the following examples: [BMI](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/generated/generated_1.ipynb), [Rational difference](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/generated/generated_2.ipynb) and [IRIS](https://github.com/ahmed-mohamed-sn/ATgfe/blob/master/examples/toy-examples/iris_multi_classification.ipynb).

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
| IRIS  (4 features)               | <ul>     <li>10-CV Accuracy: 0.926</li><li>Test-data Accuracy: 0.911</li><li>ROC_AUC: 0.99</li> </ul>        |  <ul>     <li>10-CV Accuracy: 0.946</li><li>Test-data Accuracy: 0.977</li><li>ROC_AUC: 1.0</li> </ul>    | <ul>  <li>10-CV Accuracy: **0.98**</li><li>Test-data Accuracy: **1.0**</li><li>ROC_AUC: **1.0**</li> </ul> |

## Regression
| Dataset                          | Linear Regression                                                       | LightGBM Regressor                                                     | Linear Regression + ATgfe                                          |
|----------------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------|
| Concrete  (8 features)           | <ul>     <li>10-CV RMSE: 11.13</li><li>Test-data RMSE: 10.38</li><li>r^2: 0.644</li> </ul>             | <ul>     <li>10-CV RMSE: 6.44</li><li>Test-data RMSE: **4.23**</li>     <li>r^2: **0.941**</li> </ul>     | <ul>     <li>10-CV RMSE: **6.00**</li><li>Test-data RMSE: 5.45</li>     <li>r^2: 0.899</li> </ul>         |
| Boston  (13 features)            | <ul>     <li>10-CV RMSE: 4.796</li><li>Test-data RMSE: 4.714</li><li>r^2: 0.765</li> </ul>             | <ul>     <li>10-CV RMSE: 3.38</li> <li>Test-data RMSE: 3.63</li>    <li>r^2: 0.86</li> </ul>             | <ul>  <li>10-CV RMSE: **3.125**</li><li>Test-data RMSE: **2.758**</li><li>r^2: **0.920**</li> </ul>         |

# Get started

## Requirements
- Python ^3.6
- DEAP ^1.3
- Pandas ^0.25.2
- Scipy ^1.3
- Numpy ^1.17
- Sympy ^1.4

## Install ATgfe
```bash
pip install atgfe
```
## Upgrade ATgfe
```bash
pip install -U atgfe
```
# Usage

## Examples
The [Examples](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/) are grouped under the following two sections:
- [Generated](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/generated) examples test ATgfe against hand-crafted non-linear problems where we know there is information that can be captured using feature interactions. 

- [Toy Examples](https://github.com/ahmed-mohamed-sn/ATgfe/tree/master/examples/toy-examples) show how to use ATgfe in solving a mix of regression and classification problems from publicly available benchmark datasets.

## Pre-processing for column names
### ATgfe requires column names that are free from special characters and spaces (e.g. @, $, %, #, etc.)
```python
# example
def prepare_column_names(columns):
    return [col.replace(' ', '').replace('(cm)', '_cm') for col in columns]

columns = prepare_column_names(df.columns.tolist())
df.columns = columns
```

## Configuring the parameters of GeneticFeatureEngineer
```python
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
    enable_bias: bool = False,
    max_bias: float = 100.0,
    weights_number_of_decimal_places: int = 2,
    shuffle_training_data_every_generation: bool = False,
    cross_validation_in_objective_func: bool = False,
    objective_func_cv: int = 3,
    n_jobs: int = 1,
    verbose: bool = True
)
```

### model
ATgfe works with any model or pipeline that follows scikit-learn API (i.e. the model should implement the ```fit()``` and ```predict()``` methods).

### x_train
Training features in a pandas Dataframe.

### y_train
Training labels in a pandas Dataframe to also handle multiple target problems.

### numerical_features
The list of column names that represent the numerical features.

### number_of_candidate_features
The maximum number of features to be generated.

### number_of_interacting_features
The maximum number of existing features that can be used in constructing new features. 
These features are selected from those passed in the ```numerical_features``` argument.

### evaluation_metric
Any of the [scitkit-learn metrics](https://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics) or a custom-made evaluation metric to be used by the genetic algorithm to evaluate the predictive power of the newly generated features. 
```python
import numpy as np
from sklearn.metrics import  mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```
### minimize_metric
A boolean flag, which should be set to ```True``` if the evaluation metric is to be minimized; otherwise set to ```False``` if the evaluation metric is to be maximized.
 
### categorical_features
The list of column names that represent the categorical features. The parameter ```enable_grouping``` should be set to ```True``` in order for the ```categorical_features``` to be utilized in grouping.

### enable_grouping
A boolean flag, which should be set to ```True``` to construct complex feature interactions that use ```pandas.groupBy```.

### sampling_size
The exact size of the sampled training dataset. Use this parameter to run the optimization using the specified number of observations in the training data. If the ```sampling_size``` is greater than the number of observations, then ATgfe will create a sample with replacement.
 
### cv
The number of folds for cross validation. Every generation of the genetic algorithm, ATgfe evaluates the current best solution using k-fold cross validation. The default number of folds is 10.

### fit_wo_original_columns
A boolean flag, which should be set to ```True``` to fit the model without the original features specified in ```numerical_features```. In this case, ATgfe will only use the newly generated features together with any remaining original features in ```x_train```.

### enable_feature_transformation_operations
A boolean flag, which should be set to ```True``` to enable scientific feature interactions on the ```numerical_features```.
The pre-defined transformation operators are listed as follows:
```
np_log(), np_log_10(), np_exp(), squared(), cube()
```
You can easily remove from or add to the existing list of transformation operators. Check out the next section for examples.

### enable_weights
A boolean flag, which should be set to ```True``` to enable weighted feature interactions.

### weights_number_of_decimal_places
The number of decimal places (i.e. precision) to be applied to the weight values.

### enable_bias
A boolean flag, which enables the genetic algorithm to add a bias to the expressions generated. For example:
```
0.43*log(cement) + 806.8557595548646
```

### max_bias
The value of the bias will be between ```-max_bias``` and ```max_bias```.
If the ```max_bias``` is 100 then the bias value will be between -100 and 100.

### shuffle_training_data_every_generation
A boolean flag, if enabled the ```train_test_split``` method in the objective function uses the generation number as its random seed. This can prevent over-fitting. <br/>
This option is only available if ```cross_validation_in_objective_func``` is set to ```False```.  

### cross_validation_in_objective_func
A boolean flag, if enabled the ```train_test_split``` method will not be used in the objective function. Instead of using ```train_test_split```, the genetic algorithm will use cross validation to evaluate the generated features.
<br/> The default number of folds is **3**. The number of folds can modified using the ```objective_func_cv``` parameter.

### objective_func_cv
The number of folds to be used when ```cross_validation_in_objective_func``` is enabled.

### verbose
A boolean flag, which should be set to ```True``` to enable the logging functionality.

### n_jobs
To enable parallel processing, set ```n_jobs``` to the number of CPUs that you would like to utilise. If ```n_jobs``` is set to **-1**, all the machine's CPUs will be utilised.

## Configuring the parameters of fit()
```python
gfe.fit(
    number_of_generations: int = 100,
    mu: int = 10,
    lambda_: int = 100,
    crossover_probability: float = 0.5,
    mutation_probability: float = 0.2,
    early_stopping_patience: int = 5,
    random_state: int = 77
)
```

### number_of_generations
The maximum number of generations to be explored by the genetic algorithm.

### mu
The number of solutions to select for the next generation.

### lambda_
The number of children to produce at each generation.

### crossover_probability
The crossover probability.

### mutation_probability
The mutation probability.

### early_stopping_patience
The maximum number of generations to be explored before early the stopping criteria is satisfied when the validation score is not improving.


## Configuring the parameters of transform()
```python
X = gfe.transform(X)
```

Where X is the pandas dataframe that you would like to append the generated features to.

## Transformation operations

### Get current transformation operations
```python
gfe.get_enabled_transformation_operations()
```

The enabled transformation operations will be returned.

```
['None', 'np_log', 'np_log_10', 'np_exp', 'squared', 'cube']
```
### Remove existing transformation operations
```gfe.remove_transformation_operation``` accepts string or a list of strings
```python
gfe.remove_transformation_operation('squared')
```

```python
gfe.remove_transformation_operation(['np_log_10', 'np_exp'])
```
### Add new transformation operations 
```python
np_sqrt = np.sqrt

def some_func(x):
    return (x * 2)/3

gfe.add_transformation_operation('sqrt', np_sqrt)
gfe.add_transformation_operation('some_func', some_func)
```