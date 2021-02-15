# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
url = "/home/cuip/Desktop/TAZ csv/Chattanooga_Tigerline_TAZ_Enrich.csv"
names = ['OBJECTID','STATEFP10','COUNTYFP10','MPOCE10','TADCE10','TAZCE10','GEOID10','ALAND10','AWATER10','INTPTLAT10','INTPTLON10','HasData','ORIGINAL_OID','householdincome_avghinc_cy','householdtotals_tothh_cy','vehiclesavailable_acsaggveh','populationtotals_dpop_cy','populationtotals_dpopwrk_cy','populationtotals_dpopres_cy','populationtotals_dpopdenscy','industry_unemprt_cy','vehiclesavailable_acsavgveh','Shape_Length','Shape_Area','Shape_Area_sqmi','Shape_Area_sqmi1','ppl_per_sqmi']
dataframe = read_csv(url, names=names)
numpy.nan_to_num(dataframe)
array = dataframe.values
X = array[:,0:26]
Y = array[:,:]
# feature extraction
pca = PCA(n_components=5)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)