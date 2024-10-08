# Test Cases


To evaluate the best model performance we test different hyperparams for each model. First, channels against other params were tested following the below arrangments:

## Model v4
- channels: [R, G, B] both t0 and t1
- normalized dataset between 0 - 1
- enhanced brighted - 1.5
- dataset v3
- tensorflow==2.14.0

## Model v5
- channels: [R, G, B] both t0 and t1
- normalized dataset between 0 - 1
- dataset v4
- performance: 
    - loss: 0.1580 
    - accuracy: 0.9970 
    - running_recall: 0.9031 
    - running_f1: 0.8097 
    - running_precision: 0.7370 
    - io_u: 0.4177
    - tensorflow==2.14.0

## Model v6
- channels: [ NDFI ] both t0 and t1
- normalized dataset between 0 - 1
- dataset v5
- performance: 
    - loss: 0.9927
    - accuracy: 0.9927 
    - running_recall: 0.0000e+00
    - running_f1: 0.0000e+00
    - running_precision: 0.0000e+00 
    - io_u: 0.0000e+00
    - tensorflow==2.14.0


## Model v7
- channels: [ GV, NPV, SOIL, SHADE, CLOUD, NDFI ] both t0 and t1
- normalized dataset between 0 - 1
- dataset v5
- performance: 
    - loss: 0.4664
    - accuracy: 0.9950 
    - running_recall: 0.3468
    - running_f1: 0.4328
    - running_precision: 0.6665
    - io_u: 0.1732
    - tensorflow==2.14.0

## Model v8
- channels: [ GV, NPV, SOIL, SHADE, CLOUD, NDFI ] both t0 and t1
- dataset v6
- performance: 
    - loss: 
    - accuracy:  
    - running_recall: 
    - running_f1: 
    - running_precision: 
    - io_u: 
    - tensorflow==2.16.1

## Model v9
- channels: [R, G, B] both t0 and t1
- normalized dataset between 0 - 1
- dataset v6
- performance: 
    - loss: 
    - accuracy:  
    - running_recall: 
    - running_f1: 
    - running_precision: 
    - io_u: 
    - tensorflow==2.16.1

## Dataset v3
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1

## Dataset v4
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1
- label refined with IA

## Dataset v5
- channels: ['gv','npv', 'soil', 'shade', 'cloud', 'ndfi'] both t0 and t1
- label refined with IA






## Dataset v6
- channels: ['gv','npv', 'soil', 'shade', 'cloud', 'ndfi'] both t0 and t1
- label refined with IA and fixed using roads

## Dataset v7
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1
- label refined with IA and fixed using roads

## Dataset v8
- channels: ['red','green', 'blue', 'nir', 'swir1'] both t0 and t1
- label refined with IA and fixed using roads and removed bad samples

## Dataset Bucket logging_v3
- channels: ['all'] both t0 and t1
- label refined with Roads IA


--