# Test Cases


To evaluate the best model performance we test different hyperparams for each model. First, channels against other params were tested following the below arrangments:

## Model v4
- channels: [R, G, B] both t0 and t1
- normalized dataset between 0 - 1
- enhanced brighted - 1.5
- dataset v3

## Model v5
- channels: [R, G, B] both t0 and t1
- normalized dataset between 0 - 1
- dataset v4



## Dataset v3
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1

## Dataset v4
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1
- label refined with IA

## Dataset v5
- channels: ['gv','npv', 'soil', 'shade', 'cloud', 'ndfi'] both t0 and t1
- label refined with IA