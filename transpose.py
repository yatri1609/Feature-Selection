import pandas as pd

#reads the csv 
#instead of /home.... put your path to the csv
data = pd.read_csv('/home/cuip/Desktop/TAZ csv/Chat_NLCD_Zonal_Hist_of_Dev_to_Undev.csv')
data_t = data.transpose()
print(data_t.head())

#change the name of your output file to anything you want 
data_t.to_csv('out.csv', index = True)