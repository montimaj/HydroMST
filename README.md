# Groundwater Withdrawal Prediction Using Integrated Multitemporal Remote Sensing Data Sets and Machine Learning

## Abstract
Effective monitoring of groundwater withdrawals is necessary to help mitigate the negative impacts of aquifer depletion. In this study, we develop a holistic approach that combines water balance components with a machine learning model to estimate groundwater withdrawals. We use both multitemporal satellite and modeled data from sensors that measure different components of the water balance and land use at varying spatial and temporal resolutions. These remote sensing products include evapotranspiration, precipitation, and land cover. Due to the inherent complexity of integrating these data sets and subsequently relating them to groundwater withdrawals using physical models, we apply random forests—a state of the art machine learning algorithm—to overcome such limitations. Here, we predict groundwater withdrawals per unit area over a highly monitored portion of the High Plains aquifer in the central United States at 5 km resolution for the Years 2002–2019. Our modeled withdrawals had high accuracy on both training and testing data sets (R2 ≈ 0.99 and R2 ≈ 0.93, respectively) during leave-one-out (year) cross validation with low mean absolute error (MAE) ≈ 4.31 mm and root-mean-square error (RMSE) ≈ 13.50 mm for the year 2014. Moreover, we found that even for the extreme drought year of 2012, we have a satisfactory test score (R2 ≈ 0.84) with MAE ≈ 9.72 mm and RMSE ≈ 24.17 mm. Therefore, the proposed machine learning approach should be applicable to similar regions for proactive water management practices.

![preview](Preview/HydroMST_Preview.PNG)

## Dependencies

[Anaconda](https://www.anaconda.com/products/individual) is required for installing the Python 3 packages. Once Anaconda
is installed, the following can be run from the command line to install the required packages.

```
conda env create -f environment.yml
```

We use [OSGeo4W](https://www.osgeo.org/projects/osgeo4w/) binary distribution for Windows and use system call from Python to generate rasters. For Linux or MacOS, gdal needs to be installed separately, and the appropriate 'gdal_path' needs to be set in Python_Files/gw_driver.py

## Publications
Related research article:  https://doi.org/10.1029/2020WR028059

Citation:
```
@article{https://doi.org/10.1029/2020WR028059,
author = {Majumdar, S. and Smith, R. and Butler Jr., J. J. and Lakshmi, V.},
title = {Groundwater Withdrawal Prediction Using Integrated Multitemporal Remote Sensing Data Sets and Machine Learning},
journal = {Water Resources Research},
volume = {56},
number = {11},
pages = {e2020WR028059},
keywords = {Groundwater hydrology, Remote sensing, Machine learning, Time series analysis, Estimation and forecasting, Geospatial},
doi = {https://doi.org/10.1029/2020WR028059},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020WR028059},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2020WR028059},
note = {e2020WR028059 2020WR028059},
year = {2020}
}
```

## Acknowledgments
We would like to acknowledge all the open-source software and data communities for making their resources publicly available. Particularly, we would like to thank NASA/JPL (https://grace.jpl.nasa.gov/data-analysis-tool/ and https://doi.org/10.5067/MODIS/MOD16A2.006), USDA (https://nassgeodata.gmu.edu/CropScape/), and the PRISM group (http://www.prism.oregonstate.edu/) for providing all the necessary remote sensing data sets. The work of the third author was supported, in part, by the Kansas Water Plan under the Ogallala-High Plains Aquifer Assessment Program (OHPAAP) and the United States Department of Agriculture (USDA) and the United States National Science Foundation (NSF) under USDA-NIFA/NSF INFEWS subaward RC108063UK. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the OHPAAP, USDA, or NSF. We thank Brownie Wilson of the Kansas Geological Survey for helping us access the WIMAS database (http://hercules.kgs.ku.edu/geohydro/wimas/). Finally, we are grateful to our colleagues and families for their continuous motivation and support.

<img src="Preview/MST.png" height="80"/> &nbsp; <img src="Preview/Kgs_logo_lg.png" height="80"/> &nbsp; <img src="Preview/UVA-Logo.png" height="80"/> &nbsp; <img src="Preview/usda-logo-color.png" height="80"/> &nbsp; <img src="Preview/NSF_Official_logo.png" height="100"/>