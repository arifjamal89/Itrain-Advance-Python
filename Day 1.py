#Import csv file
import pandas as pd
df1=pd.read_csv('C:/Users/arun/Desktop/ITRAIN/itrain python/Advanced/codes/DAY 1/Titanic.csv')
df1
#Check for na values
df1.isnull()
#Count the number of null values
df1.isnull().sum()
#Drop columns with NA values(when axis =1)
#df1.dropna(axis=1)
#Remove duplicates
#df1.duplicated()
#Sort the columns 
#df1.sort_values([by='colname'])
#Fill NA values as 0
df1.fillna(0)

#Describe the datset
df1.describe()
#cov
#df1.cov()



#Read text file
import pandas as pd
X = pd.read_csv('C:/Users/arun/Desktop/ITRAIN/itrain python/Advanced/codes/DAY 1/WA_Fn-UseC_-Sales-Win-Loss.txt', sep=",", header=None)
X



#Read pdf file
import PyPDF2
pdfFileObj = open('C:/Users/arun/Desktop/ITRAIN/itrain python/Advanced/codes/DAY 1/trade_report.pdf', 'rb')
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pdfReader.numPages
pageObj = pdfReader.getPage(0)
pageObj

#Expand the the output display to see more columns 
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)