# Cancer_relapse_prediction
predicting the possibility of cancer relapse from gene expression features. 

We use gene expression data from to predict if the cancer is going to relapse or not. Data files are in /data/. 
use datareader.py to load the data and use your models to compare with our results. 
The following models were tesed:<br />
1- Support vector machines with different kernels. <br />
2- Logisitc Regression. <br />
3- Random Forests. <br />
4- AdaBoost.<br />
5- XGBoost.<br />
6- Gaussian Processes with different Kernels. <br />
7- Naive bayes <br />
8- PCA <br />
9- Extra tree <br />
and a voting classifier was chosen in the end.

we use different feature selection techniques including: 
Mutual information feature selection. 
Correlation matrix. 
Variannce threshold. 
