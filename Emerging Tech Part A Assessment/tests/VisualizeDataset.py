import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_load(file_name):
    return pd.read_excel(file_name)

def Plot_All_Variables(data):
    return sns.pairplot(data)

if __name__ == "__main__":
    data = data_load('Net_Worth_Data.xlsx')
    Plot_All_Variables(data)
    plt.show()

    
    