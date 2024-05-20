# Overview

This is a short python script that uses SAGA lib to compute Topographic Wetness Index (TWI). Below you will find the steps to install the dependecies and run.

## Install SAGA
it is necessary to install SAGA GIS software, you may access the installer at the [link](https://sourceforge.net/projects/saga-gis/)

## Create and activate python virtual env
At the root dir of this repository run the code below
```
# create virtual env
python -m venv venv

# activate venv (linux)
source venv/bin/activate

# activate venv (windows cmd)
venv/Scripts/activate
```

## Install dependencies
```
pip install -r requirements.txt

```

## Run Script
Download your DEM files and put them at the path *02_saga_gis/data/input*. Your outputs will be at *02_saga_gis/data/output*

