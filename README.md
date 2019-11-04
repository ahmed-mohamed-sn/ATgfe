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