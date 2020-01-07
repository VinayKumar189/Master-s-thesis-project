#importing required libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import metrics 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


# function to plot scattered line 
def scatter_plot_with_correlation_line(x, y, graph_filepath):
    plt.scatter(x, y)
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.xlabel("Calculated Cohesive energy (eV/atom)")
    plt.ylabel("Predicted Cohesive energy (eV/atom)")
    plt.plot(X_plot, m*X_plot + b, '-')
    plt.savefig(graph_filepath, dpi=1000, format='png', bbox_inches='tight')
    plt.show()

# reading features from csv file
df = pd.read_csv('newfeatures_data.csv', sep=',')
print('df:','\n',df)

# seperating features based on type and property
columns_Eng = ['ionenergies[1]_1','ionenergies[1]_2','ionenergies[1]_3','ionenergies[1]_4','ionenergies[1]_5','electron_affinity_1','electron_affinity_2','electron_affinity_3','electron_affinity_4','electron_affinity_5']
columns_dim_less = ['en_pauling_1','en_pauling_2','en_pauling_3','en_pauling_4','en_pauling_5','atomic_number_1','atomic_number_2','atomic_number_3','atomic_number_4','atomic_number_5','period_1','period_2','period_3','period_4','period_5','group_id_1','group_id_2','group_id_3','group_id_4','group_id_5',]
columns_dist = ['covalent_radius_bragg_1','covalent_radius_bragg_2','covalent_radius_bragg_3','covalent_radius_bragg_4','covalent_radius_bragg_5','atomic_volume_1','atomic_volume_2','atomic_volume_3','atomic_volume_4','atomic_volume_5']

df_Eng = df[columns_Eng]
df_Eng_3 = df_Eng[0:51]
print('df_Eng_3:','\n',df_Eng_3)
df_Eng_4 = df_Eng[51:175]
print('df_Eng_4:','\n',df_Eng_4)
df_Eng_5 = df_Eng[175:205]
print('df_Eng_5:','\n',df_Eng_5)
df_dl  = df[columns_dim_less]
df_dl_3 = df_dl[0:51]
print(df_dl_3)
df_dl_4 = df_dl[51:175]
print(df_dl_4)
df_dl_5 = df_dl[175:205]
print(df_dl_5)
df_dt  = df[columns_dist]
df_dt_3 = df_dt[0:51]
print(df_dt_3)
df_dt_4 = df_dt[51:175]
print(df_dt_4)
df_dt_5 = df_dt[175:205]
print(df_dt_5)
print('df_Eng:', '\n', df_Eng,'\n','df_dl:','\n',df_dl,'\n','df_dt:','\n',df_dt)

# appling non_linear functions on above features
# exponentianl function
df_exp = pd.DataFrame()
def mapper(name):
    return name + '_exp'

for i in range(3,6):
    if i == 3:  
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
       df_new_i = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()
 
    for j in Eng_i:
        if i == 3: 
            x = pd.Series(df_Eng_3[j])
            if x.any() == True: 
                 df_new_i[j] =  df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.exp(x))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.exp(x))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.exp(x))

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.exp(x))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.exp(x))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.exp(x))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.exp(x))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.exp(x))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.exp(x))
                
    print('df_new:', df_new_i) 
    df_exp = df_exp.append(df_new_i)

df_exp.rename(mapper=mapper, axis=1, inplace=True)
df_exp.fillna(0, inplace=True)
print('df_exp:', df_exp)

# log function
df_log = pd.DataFrame()
def mapper(name):
    return name + '_log'

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
       df_e_3 = pd.DataFrame()
       df_dim_3 = pd.DataFrame()
       df_dis_3 = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
       df_new_i = pd.DataFrame()
       df_e_4  = pd.DataFrame()
       df_dim_4 = pd.DataFrame()
       df_dis_4 = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()
       df_e_5 = pd.DataFrame()
       df_dim_5 = pd.DataFrame()
       df_dis_5 = pd.DataFrame()

    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_3[j]
                 df_e_3[j] = df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.log(x))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 df_e_4[j] = df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.log(x))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 df_e_5[j] = df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.log(x))
    
    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                df_dim_3[k] = df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.log(x))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                df_dim_4[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.log(x))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                df_dim_5[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.log(x))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              df_dis_3[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.log(x))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               df_dis_4[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.log(x))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               df_dis_5[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.log(x))

    print('df_new:', df_new_i)
    df_log = df_log.append(df_new_i)

df_log.rename(mapper=mapper, axis=1, inplace=True)
df_log.fillna(0, inplace=True)
print('df_log:', df_log)
print("df_e_3:", df_e_3)
print("df_dim_4:", df_dim_4)

# square root function
df_sqrt = pd.DataFrame()
def mapper(name):
    return name + '_sqrt'

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
       df_new_i = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()

    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.sqrt(x))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.sqrt(x))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.sqrt(x))

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.sqrt(x))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.sqrt(x))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.sqrt(x))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.sqrt(x))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.sqrt(x))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.sqrt(x))

    print('df_new:', df_new_i)
    df_sqrt = df_sqrt.append(df_new_i)

df_sqrt.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt.fillna(0, inplace=True)
print('df_sqrt:', df_sqrt)


# square function
df_sq = pd.DataFrame()
def mapper(name):
    return name + '_sq'

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
       df_new_i = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()

    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,2))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,2))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,2))

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,2))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,2))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,2))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,2))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,2))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,2))

    print('df_new:', df_new_i)
    df_sq = df_sq.append(df_new_i)

df_sq.rename(mapper=mapper, axis=1, inplace=True)
df_sq.fillna(0, inplace=True)
print('df_sq:', df_sq)

# cube function
df_cu = pd.DataFrame()
def mapper(name):
    return name + '_cu'

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
       df_new_i = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()

    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,3))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,3))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.power(x,3))

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,3))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,3))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.power(x,3))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,3))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,3))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.power(x,3))

    print('df_new:', df_new_i)
    df_cu = df_cu.append(df_new_i)

df_cu.rename(mapper=mapper, axis=1, inplace=True)
df_cu.fillna(0, inplace=True)
print('df_cu:', df_cu)


# reciprocal function
df_rp = pd.DataFrame()
def mapper(name):
    return name + '_rp'

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
       df_new_i = pd.DataFrame()
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       df_new_i = pd.DataFrame()
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
       df_new_i = pd.DataFrame()

    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_3[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_4[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 df_new_i[j] =  df_Eng_5[j]
                 print(df_new_i[j])
                 df_new_i[j] = df_new_i[j].apply(lambda x: np.float_power(x,-1))

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                df_new_i[k] =  df_dl_3[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_4[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                df_new_i[k] =  df_dl_5[k]
                print(df_new_i[k])
                df_new_i[k] = df_new_i[k].apply(lambda x: np.float_power(x,-1))

    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              df_new_i[l] =  df_dt_3[l]
              print(df_new_i[l])
              df_new_i[l] = df_new_i[l].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_4[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               df_new_i[l] =  df_dt_5[l]
               print(df_new_i[l])
               df_new_i[l] = df_new_i[l].apply(lambda x: np.float_power(x,-1))

    print('df_new:', df_new_i)
    df_rp = df_rp.append(df_new_i)

df_rp.rename(mapper=mapper, axis=1, inplace=True)
df_rp.fillna(0, inplace=True)
print('df_rp:', df_rp)

# appling linear combinations on non-linear features
e_3 = []
e_4 = []
e_5 = []
dim_3 = []
dim_4 = []
dim_5 = []
dis_3 = []
dis_4 = []
dis_5 = []

for i in range(3,6):
    if i == 3:
       Eng_i = list(df_Eng_3)
       dl_i = list(df_dl_3)
       dt_i = list(df_dt_3)
    elif i == 4:
       Eng_i = list(df_Eng_4)
       dl_i = list(df_dl_4)
       dt_i = list(df_dt_4)
    elif i == 5:
       Eng_i = list(df_Eng_5)
       dl_i = list(df_dl_5)
       dt_i = list(df_dt_5)
    
    for j in Eng_i:
        if i == 3:
            x = pd.Series(df_Eng_3[j])
            if x.any() == True:
                 e_3.append(j)
        elif i == 4:
            x = pd.Series(df_Eng_4[j])
            if x.any() == True:
                 e_4.append(j)
        elif i == 5:
            x = pd.Series(df_Eng_5[j])
            if x.any() == True:
                 e_5.append(j)

    for k in dl_i:
        if i == 3:
           y = pd.Series(df_dl_3[k])
           if y.any() == True:
                dim_3.append(k)
        elif i == 4:
            y = pd.Series(df_dl_4[k])
            if y.any() == True:
                dim_4.append(k)
        elif i == 5:
            y = pd.Series(df_dl_5[k])
            if y.any() == True:
                dim_5.append(k)
     
    for l in dt_i:
        if i == 3:
           z = pd.Series(df_dt_3[l])
           if z.any() == True:
              dis_3.append(l)
        elif i == 4:
           z = pd.Series(df_dt_4[l])
           if z.any() == True:
               dis_4.append(l)
        elif i == 5:
           z = pd.Series(df_dt_5[l])
           if z.any() == True:
               dis_5.append(l)
     
for i in range(3,6):
    if i == 3:
        print("e_3:", e_3)
        print("dim_3:", dim_3)
        print("dis_3:", dis_3)
    elif i == 4:
        print("e_4:", e_4)
        print("dim_4:", dim_4)
        print("dis_4:", dis_4)
    elif i == 5:
        print("e_5:", e_5)
        print("dim_5:", dim_5)
        print("dis_5:", dis_5)

df_e_bin_3 = pd.DataFrame()
for i in range(len(e_3)):
    if i < (len(e_3)-1):
        df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+1]] = df_e_3[e_3[i]] + df_e_3[e_3[i+1]]
        df_e_bin_3["|"+e_3[i]+" "+"+"+" "+e_3[i+1]+"|"] = df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+1]].abs()
        df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+1]] = df_e_3[e_3[i]] - df_e_3[e_3[i+1]]
        df_e_bin_3["|"+e_3[i]+" "+"-"+" "+e_3[i+1]+"|"] = df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+1]].abs()
       
    if i < (len(e_3)-2):
        df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+2]] = df_e_3[e_3[i]] + df_e_3[e_3[i+2]]
        df_e_bin_3["|"+e_3[i]+" "+"+"+" "+e_3[i+2]+"|"] = df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+2]].abs()
        df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+2]] = df_e_3[e_3[i]] - df_e_3[e_3[i+2]]
        df_e_bin_3["|"+e_3[i]+" "+"-"+" "+e_3[i+2]+"|"] = df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+2]].abs()
    
    if i < (len(e_3)-3):
        df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+3]] = df_e_3[e_3[i]] + df_e_3[e_3[i+3]]
        df_e_bin_3["|"+e_3[i]+" "+"+"+" "+e_3[i+3]+"|"] = df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+3]].abs()
        df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+3]] = df_e_3[e_3[i]] - df_e_3[e_3[i+3]]
        df_e_bin_3["|"+e_3[i]+" "+"-"+" "+e_3[i+3]+"|"] = df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+3]].abs()
        
    if i < (len(e_3)-4):
        df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+4]] = df_e_3[e_3[i]] + df_e_3[e_3[i+4]]
        df_e_bin_3["|"+e_3[i]+" "+"+"+" "+e_3[i+4]+"|"] = df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+4]].abs()
        df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+4]] = df_e_3[e_3[i]] - df_e_3[e_3[i+4]]
        df_e_bin_3["|"+e_3[i]+" "+"-"+" "+e_3[i+4]+"|"] = df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+4]].abs()
        
    if i < (len(e_3)-5):
        df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+5]] = df_e_3[e_3[i]] + df_e_3[e_3[i+5]]
        df_e_bin_3["|"+e_3[i]+" "+"+"+" "+e_3[i+5]+"|"] = df_e_bin_3[e_3[i]+" "+"+"+" "+e_3[i+5]].abs()
        df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+5]] = df_e_3[e_3[i]] - df_e_3[e_3[i+5]]
        df_e_bin_3["|"+e_3[i]+" "+"-"+" "+e_3[i+5]+"|"] = df_e_bin_3[e_3[i]+" "+"-"+" "+e_3[i+5]].abs()
    


print(df_e_bin_3)
print(list(df_e_bin_3))

df_e_bin_4 = pd.DataFrame()
for i in range(len(e_4)):
    if i < (len(e_4)-1):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+1]] = df_e_4[e_4[i]] + df_e_4[e_4[i+1]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+1]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+1]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+1]] = df_e_4[e_4[i]] - df_e_4[e_4[i+1]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+1]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+1]].abs()

    if i < (len(e_4)-2):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+2]] = df_e_4[e_4[i]] + df_e_4[e_4[i+2]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+2]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+2]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+2]] = df_e_4[e_4[i]] - df_e_4[e_4[i+2]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+2]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+2]].abs()

    if i < (len(e_4)-3):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+3]] = df_e_4[e_4[i]] + df_e_4[e_4[i+3]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+3]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+3]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+3]] = df_e_4[e_4[i]] - df_e_4[e_4[i+3]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+3]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+3]].abs()

    if i < (len(e_4)-4):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+4]] = df_e_4[e_4[i]] + df_e_4[e_4[i+4]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+4]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+4]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+4]] = df_e_4[e_4[i]] - df_e_4[e_4[i+4]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+4]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+4]].abs()

    if i < (len(e_4)-5):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+5]] = df_e_4[e_4[i]] + df_e_4[e_4[i+5]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+5]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+5]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+5]] = df_e_4[e_4[i]] - df_e_4[e_4[i+5]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+5]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+5]].abs()

    if i < (len(e_4)-6):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+6]] = df_e_4[e_4[i]] + df_e_4[e_4[i+6]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+6]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+6]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+6]] = df_e_4[e_4[i]] - df_e_4[e_4[i+6]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+6]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+6]].abs()
        
    if i < (len(e_4)-7):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+7]] = df_e_4[e_4[i]] + df_e_4[e_4[i+7]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+7]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+7]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+7]] = df_e_4[e_4[i]] - df_e_4[e_4[i+7]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+7]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+7]].abs()

print(df_e_bin_4)
print(list(df_e_bin_4))

        
df_e_bin_5 = pd.DataFrame()
for i in range(len(e_5)):
    if i < (len(e_5)-1):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+1]] = df_e_5[e_5[i]] + df_e_5[e_5[i+1]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+1]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+1]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+1]] = df_e_5[e_5[i]] - df_e_5[e_5[i+1]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+1]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+1]].abs()

    if i < (len(e_5)-2):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+2]] = df_e_5[e_5[i]] + df_e_5[e_5[i+2]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+2]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+2]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+2]] = df_e_5[e_5[i]] - df_e_5[e_5[i+2]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+2]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+2]].abs()

    if i < (len(e_5)-3):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+3]] = df_e_5[e_5[i]] + df_e_5[e_5[i+3]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+3]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+3]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+3]] = df_e_5[e_5[i]] - df_e_5[e_5[i+3]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+3]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+3]].abs()

    if i < (len(e_5)-4):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+4]] = df_e_5[e_5[i]] + df_e_5[e_5[i+4]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+4]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+4]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+4]] = df_e_5[e_5[i]] - df_e_5[e_5[i+4]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+4]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+4]].abs()

    if i < (len(e_5)-5):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+5]] = df_e_5[e_5[i]] + df_e_5[e_5[i+5]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+5]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+5]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+5]] = df_e_5[e_5[i]] - df_e_5[e_5[i+5]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+5]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+5]].abs()

    if i < (len(e_5)-6):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+6]] = df_e_5[e_5[i]] + df_e_5[e_5[i+6]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+6]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+6]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+6]] = df_e_5[e_5[i]] - df_e_5[e_5[i+6]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+6]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+6]].abs()
    
    if i < (len(e_5)-7):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+7]] = df_e_5[e_5[i]] + df_e_5[e_5[i+7]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+7]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+7]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+7]] = df_e_5[e_5[i]] - df_e_5[e_5[i+7]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+7]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+7]].abs()

    if i < (len(e_5)-8):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+8]] = df_e_5[e_5[i]] + df_e_5[e_5[i+8]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+8]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+8]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+8]] = df_e_5[e_5[i]] - df_e_5[e_5[i+8]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+8]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+8]].abs()

    if i < (len(e_5)-9):
        df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+9]] = df_e_5[e_5[i]] + df_e_5[e_5[i+9]]
        df_e_bin_5["|"+e_5[i]+" "+"+"+" "+e_5[i+9]+"|"] = df_e_bin_5[e_5[i]+" "+"+"+" "+e_5[i+9]].abs()
        df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+9]] = df_e_5[e_5[i]] - df_e_5[e_5[i+9]]
        df_e_bin_5["|"+e_5[i]+" "+"-"+" "+e_5[i+9]+"|"] = df_e_bin_5[e_5[i]+" "+"-"+" "+e_5[i+9]].abs()
        
print(df_e_bin_5)
print(list(df_e_bin_5))

print(len(list(dim_3)))
print(len(list(dim_4)))

print(len(list(dis_3)))
print(len(list(dis_4)))


df_dim_bin_3 = pd.DataFrame()
for i in range(len(dim_3)):
    if i < (len(dim_3)-1):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+1]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+1]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+1]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+1]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+1]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+1]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+1]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+1]].abs()

    if i < (len(dim_3)-2):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+2]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+2]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+2]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+2]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+2]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+2]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+2]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+2]].abs()

    if i < (len(dim_3)-3):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+3]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+3]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+3]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+3]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+3]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+3]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+3]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+3]].abs()

    if i < (len(dim_3)-4):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+4]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+4]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+4]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+4]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+4]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+4]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+4]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+4]].abs()

    if i < (len(dim_3)-5):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+5]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+5]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+5]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+5]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+5]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+5]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+5]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+5]].abs()

    if i < (len(dim_3)-6):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+6]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+6]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+6]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+6]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+6]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+6]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+6]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+6]].abs()

    if i < (len(dim_3)-7):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+7]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+7]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+7]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+7]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+7]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+7]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+7]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+7]].abs()

    if i < (len(dim_3)-8):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+8]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+8]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+8]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+8]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+8]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+8]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+8]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+8]].abs()

    if i < (len(dim_3)-9):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+9]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+9]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+9]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+9]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+9]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+9]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+9]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+9]].abs()

    if i < (len(dim_3)-10):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+10]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+10]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+10]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+10]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+10]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+10]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+10]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+10]].abs()

    if i < (len(dim_3)-11):
        df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+11]] = df_dim_3[dim_3[i]] + df_dim_3[dim_3[i+11]]
        df_dim_bin_3["|"+dim_3[i]+" "+"+"+" "+dim_3[i+11]+"|"] = df_dim_bin_3[dim_3[i]+" "+"+"+" "+dim_3[i+11]].abs()
        df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+11]] = df_dim_3[dim_3[i]] - df_dim_3[dim_3[i+11]]
        df_dim_bin_3["|"+dim_3[i]+" "+"-"+" "+dim_3[i+11]+"|"] = df_dim_bin_3[dim_3[i]+" "+"-"+" "+dim_3[i+11]].abs()
        

print(df_dim_bin_3)
print(len(list(df_dim_bin_3)))

df_dim_bin_4 = pd.DataFrame()
for i in range(len(dim_4)):
    if i < (len(dim_4)-1):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+1]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+1]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+1]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+1]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+1]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+1]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+1]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+1]].abs()

    if i < (len(dim_4)-2):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+2]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+2]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+2]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+2]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+2]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+2]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+2]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+2]].abs()

    if i < (len(dim_4)-3):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+3]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+3]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+3]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+3]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+3]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+3]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+3]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+3]].abs()

    if i < (len(dim_4)-4):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+4]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+4]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+4]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+4]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+4]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+4]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+4]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+4]].abs()

    if i < (len(dim_4)-5):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+5]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+5]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+5]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+5]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+5]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+5]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+5]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+5]].abs()
        
    if i < (len(dim_4)-6):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+6]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+6]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+6]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+6]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+6]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+6]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+6]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+6]].abs()

    if i < (len(dim_4)-7):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+7]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+7]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+7]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+7]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+7]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+7]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+7]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+7]].abs()

    if i < (len(dim_4)-8):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+8]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+8]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+8]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+8]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+8]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+8]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+8]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+8]].abs()

    if i < (len(dim_4)-9):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+9]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+9]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+9]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+9]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+9]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+9]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+9]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+9]].abs()

    if i < (len(dim_4)-10):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+10]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+10]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+10]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+10]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+10]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+10]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+10]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+10]].abs()

    if i < (len(dim_4)-11):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+11]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+11]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+11]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+11]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+11]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+11]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+11]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+11]].abs()
        
    if i < (len(dim_4)-12):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+12]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+12]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+12]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+12]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+12]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+12]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+12]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+12]].abs()


    if i < (len(dim_4)-13):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+13]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+13]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+13]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+13]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+13]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+13]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+13]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+13]].abs()

    if i < (len(dim_4)-14):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+14]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+14]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+14]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+14]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+14]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+14]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+14]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+14]].abs()

    if i < (len(dim_4)-15):
        df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+15]] = df_dim_4[dim_4[i]] + df_dim_4[dim_4[i+15]]
        df_dim_bin_4["|"+dim_4[i]+" "+"+"+" "+dim_4[i+15]+"|"] = df_dim_bin_4[dim_4[i]+" "+"+"+" "+dim_4[i+15]].abs()
        df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+15]] = df_dim_4[dim_4[i]] - df_dim_4[dim_4[i+15]]
        df_dim_bin_4["|"+dim_4[i]+" "+"-"+" "+dim_4[i+15]+"|"] = df_dim_bin_4[dim_4[i]+" "+"-"+" "+dim_4[i+15]].abs()
        

print(df_dim_bin_4)
print(len(list(df_dim_bin_4)))

df_dim_bin_5 = pd.DataFrame()
for i in range(len(dim_5)):
    if i < (len(dim_5)-1):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+1]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+1]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+1]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+1]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+1]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+1]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+1]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+1]].abs()

    if i < (len(dim_5)-2):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+2]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+2]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+2]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+2]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+2]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+2]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+2]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+2]].abs()

    if i < (len(dim_5)-3):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+3]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+3]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+3]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+3]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+3]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+3]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+3]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+3]].abs()

    if i < (len(dim_5)-4):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+4]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+4]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+4]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+4]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+4]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+4]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+4]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+4]].abs()

    if i < (len(dim_5)-5):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+5]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+5]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+5]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+5]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+5]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+5]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+5]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+5]].abs()

    if i < (len(dim_5)-6):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+6]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+6]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+6]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+6]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+6]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+6]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+6]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+6]].abs()

    if i < (len(dim_5)-7):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+7]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+7]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+7]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+7]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+7]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+7]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+7]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+7]].abs()

    if i < (len(dim_5)-8):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+8]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+8]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+8]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+8]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+8]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+8]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+8]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+8]].abs()

    if i < (len(dim_5)-9):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+9]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+9]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+9]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+9]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+9]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+9]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+9]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+9]].abs()

    if i < (len(dim_5)-10):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+10]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+10]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+10]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+10]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+10]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+10]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+10]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+10]].abs()

    if i < (len(dim_5)-11):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+11]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+11]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+11]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+11]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+11]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+11]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+11]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+11]].abs()

    if i < (len(dim_5)-12):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+12]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+12]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+12]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+12]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+12]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+12]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+12]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+12]].abs()

    if i < (len(dim_5)-13):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+13]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+13]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+13]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+13]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+13]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+13]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+13]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+13]].abs()

    if i < (len(dim_5)-14):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+14]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+14]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+14]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+14]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+14]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+14]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+14]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+14]].abs()

    if i < (len(dim_5)-15):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+15]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+15]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+15]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+15]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+15]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+15]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+15]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+15]].abs()

    if i < (len(dim_5)-16):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+16]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+16]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+16]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+16]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+16]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+16]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+16]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+16]].abs()

    if i < (len(dim_5)-17):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+17]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+17]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+7]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+7]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+17]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+17]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+17]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+17]].abs()

    if i < (len(dim_5)-18):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+18]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+18]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+18]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+18]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+18]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+18]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+18]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+18]].abs()

    if i < (len(dim_5)-19):
        df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+19]] = df_dim_5[dim_5[i]] + df_dim_5[dim_5[i+19]]
        df_dim_bin_5["|"+dim_5[i]+" "+"+"+" "+dim_5[i+19]+"|"] = df_dim_bin_5[dim_5[i]+" "+"+"+" "+dim_5[i+19]].abs()
        df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+19]] = df_dim_5[dim_5[i]] - df_dim_5[dim_5[i+19]]
        df_dim_bin_5["|"+dim_5[i]+" "+"-"+" "+dim_5[i+19]+"|"] = df_dim_bin_5[dim_5[i]+" "+"-"+" "+dim_5[i+19]].abs()

print(df_dim_bin_5)
print(len(list(df_dim_bin_5)))


df_dis_bin_3 = pd.DataFrame()
for i in range(len(dis_3)):
    if i < (len(dis_3)-1):
        df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+1]] = df_dis_3[dis_3[i]] + df_dis_3[dis_3[i+1]]
        df_dis_bin_3["|"+dis_3[i]+" "+"+"+" "+dis_3[i+1]+"|"] = df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+1]].abs()
        df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+1]] = df_dis_3[dis_3[i]] - df_dis_3[dis_3[i+1]]
        df_dis_bin_3["|"+dis_3[i]+" "+"-"+" "+dis_3[i+1]+"|"] = df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+1]].abs()

    if i < (len(dis_3)-2):
        df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+2]] = df_dis_3[dis_3[i]] + df_dis_3[dis_3[i+2]]
        df_dis_bin_3["|"+dis_3[i]+" "+"+"+" "+dis_3[i+2]+"|"] = df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+2]].abs()
        df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+2]] = df_dis_3[dis_3[i]] - df_dis_3[dis_3[i+2]]
        df_dis_bin_3["|"+dis_3[i]+" "+"-"+" "+dis_3[i+2]+"|"] = df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+2]].abs()

    if i < (len(dis_3)-3):
        df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+3]] = df_dis_3[dis_3[i]] + df_dis_3[dis_3[i+3]]
        df_dis_bin_3["|"+dis_3[i]+" "+"+"+" "+dis_3[i+3]+"|"] = df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+3]].abs()
        df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+3]] = df_dis_3[dis_3[i]] - df_dis_3[dis_3[i+3]]
        df_dis_bin_3["|"+dis_3[i]+" "+"-"+" "+dis_3[i+3]+"|"] = df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+3]].abs()

    if i < (len(e_3)-4):
        df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+4]] = df_dis_3[dis_3[i]] + df_dis_3[dis_3[i+4]]
        df_dis_bin_3["|"+dis_3[i]+" "+"+"+" "+dis_3[i+4]+"|"] = df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+4]].abs()
        df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+4]] = df_dis_3[dis_3[i]] - df_dis_3[dis_3[i+4]]
        df_dis_bin_3["|"+dis_3[i]+" "+"-"+" "+dis_3[i+4]+"|"] = df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+4]].abs()

    if i < (len(dis_3)-5):
        df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+5]] = df_dis_3[dis_3[i]] + df_dis_3[dis_3[i+5]]
        df_dis_bin_3["|"+dis_3[i]+" "+"+"+" "+dis_3[i+5]+"|"] = df_dis_bin_3[dis_3[i]+" "+"+"+" "+dis_3[i+5]].abs()
        df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+5]] = df_dis_3[dis_3[i]] - df_dis_3[dis_3[i+5]]
        df_dis_bin_3["|"+dis_3[i]+" "+"-"+" "+dis_3[i+5]+"|"] = df_dis_bin_3[dis_3[i]+" "+"-"+" "+dis_3[i+5]].abs()


print(df_dis_bin_3)
print(len(list(df_dis_bin_3)))


df_dis_bin_4 = pd.DataFrame()
for i in range(len(dis_4)):
    if i < (len(dis_4)-1):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+1]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+1]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+1]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+1]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+1]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+1]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+1]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+1]].abs()

    if i < (len(dis_4)-2):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+2]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+2]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+2]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+2]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+2]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+2]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+2]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+2]].abs()

    if i < (len(dis_4)-3):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+3]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+3]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+3]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+3]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+3]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+3]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+3]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+3]].abs()

    if i < (len(dis_4)-4):
        df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+4]] = df_e_4[e_4[i]] + df_e_4[e_4[i+4]]
        df_e_bin_4["|"+e_4[i]+" "+"+"+" "+e_4[i+4]+"|"] = df_e_bin_4[e_4[i]+" "+"+"+" "+e_4[i+4]].abs()
        df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+4]] = df_e_4[e_4[i]] - df_e_4[e_4[i+4]]
        df_e_bin_4["|"+e_4[i]+" "+"-"+" "+e_4[i+4]+"|"] = df_e_bin_4[e_4[i]+" "+"-"+" "+e_4[i+4]].abs()

    if i < (len(dis_4)-5):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+5]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+5]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+5]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+5]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+5]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+5]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+5]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+5]].abs()

    if i < (len(dis_4)-6):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+6]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+6]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+6]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+6]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+6]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+6]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+6]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+6]].abs()

    if i < (len(dis_4)-7):
        df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+7]] = df_dis_4[dis_4[i]] + df_dis_4[dis_4[i+7]]
        df_dis_bin_4["|"+dis_4[i]+" "+"+"+" "+dis_4[i+7]+"|"] = df_dis_bin_4[dis_4[i]+" "+"+"+" "+dis_4[i+7]].abs()
        df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+7]] = df_dis_4[dis_4[i]] - df_dis_4[dis_4[i+7]]
        df_dis_bin_4["|"+dis_4[i]+" "+"-"+" "+dis_4[i+7]+"|"] = df_dis_bin_4[dis_4[i]+" "+"-"+" "+dis_4[i+7]].abs()

print(df_dis_bin_4)
print(len(list(df_dis_bin_4)))


df_dis_bin_5 = pd.DataFrame()
for i in range(len(dis_5)):
    if i < (len(dis_5)-1):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+1]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+1]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+1]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+1]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+1]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+1]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+1]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+1]].abs()

    if i < (len(dis_5)-2):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+2]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+2]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+2]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+2]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+2]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+2]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+2]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+2]].abs()

    if i < (len(dis_5)-3):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+3]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+3]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+3]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+3]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+3]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+3]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+3]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+3]].abs()

    if i < (len(dis_5)-4):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+4]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+4]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+4]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+4]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+4]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+4]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+4]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+4]].abs()

    if i < (len(dis_5)-5):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+5]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+5]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+5]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+5]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+5]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+5]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+5]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+5]].abs()

    if i < (len(dis_5)-6):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+6]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+6]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+6]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+6]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+6]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+6]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+6]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+6]].abs()

    if i < (len(dis_5)-7):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+7]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+7]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+7]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+7]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+7]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+7]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+7]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+7]].abs()

    if i < (len(dis_5)-8):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+8]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+8]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+8]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+8]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+8]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+8]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+8]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+8]].abs()

    if i < (len(dis_5)-9):
        df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+9]] = df_dis_5[dis_5[i]] + df_dis_5[dis_5[i+9]]
        df_dis_bin_5["|"+dis_5[i]+" "+"+"+" "+dis_5[i+9]+"|"] = df_dis_bin_5[dis_5[i]+" "+"+"+" "+dis_5[i+9]].abs()
        df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+9]] = df_dis_5[dis_5[i]] - df_dis_5[dis_5[i+9]]
        df_dis_bin_5["|"+dis_5[i]+" "+"-"+" "+dis_5[i+9]+"|"] = df_dis_bin_5[dis_5[i]+" "+"-"+" "+dis_5[i+9]].abs()

print(df_dis_bin_5)

# appling non-linear functions on binary features
# exponentianl function on binary
df_exp_bin = pd.DataFrame()
def mapper(name):
    return name + '_exp'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_new[l] = df_new[l].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_new[l] = df_new[l].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_new[l] = df_new[l].apply(lambda x: np.exp(x))

    print('df_new:', df_new)
    df_exp_bin = df_exp_bin.append(df_new)

df_exp_bin.rename(mapper=mapper, axis=1, inplace=True)
df_exp_bin.fillna(0, inplace=True)
print('df_exp_bin:', df_exp_bin)


# log function on binary
df_log_bin = pd.DataFrame()
def mapper(name):
    return name + '_log'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_new[j] = df_new[j].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_new[j] = df_new[j].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_new[j] = df_new[j].apply(lambda x: np.log(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_new[k] = df_new[k].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_new[k] = df_new[k].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_new[k] = df_new[k].apply(lambda x: np.log(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_new[l] = df_new[l].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_new[l] = df_new[l].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_new[l] = df_new[l].apply(lambda x: np.log(x))

    print('df_new:', df_new)
    df_log_bin = df_log_bin.append(df_new)

df_log_bin.rename(mapper=mapper, axis=1, inplace=True)
df_log_bin.fillna(0, inplace=True)
print('df_log_bin:', df_log_bin)


# square root function on binary
df_sqrt_bin = pd.DataFrame()
df_sqrt_eng_3 = pd.DataFrame()
df_sqrt_eng_4 = pd.DataFrame()
df_sqrt_eng_5 = pd.DataFrame()
df_sqrt_dt_3 = pd.DataFrame()
df_sqrt_dt_4 = pd.DataFrame()
df_sqrt_dt_5 = pd.DataFrame()
df_sqrt_dl_3 = pd.DataFrame()
df_sqrt_dl_4 = pd.DataFrame()
df_sqrt_dl_5 = pd.DataFrame()

def mapper(name):
    return name + '_sqrt'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_sqrt_eng_3[j] = df_new[j].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_sqrt_eng_4[j] = df_new[j].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_sqrt_eng_5[j] = df_new[j].apply(lambda x: np.sqrt(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_sqrt_dl_3[k] = df_new[k].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_sqrt_dl_4[k] = df_new[k].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_sqrt_dl_5[k] = df_new[k].apply(lambda x: np.sqrt(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_sqrt_dt_3[l] = df_new[l].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_sqrt_dt_4[l] = df_new[l].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_sqrt_dt_5[l] = df_new[l].apply(lambda x: np.sqrt(x))

df_sqrt_eng_3.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_eng_4.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_eng_5.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dt_3.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dt_4.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dt_5.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dl_3.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dl_4.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_dl_5.rename(mapper=mapper, axis=1, inplace=True)

frames_eng = [df_sqrt_eng_3, df_sqrt_eng_4, df_sqrt_eng_5,]
frames_dt = [df_sqrt_dt_3, df_sqrt_dt_4, df_sqrt_dt_5]
frames_dl = [df_sqrt_dl_3, df_sqrt_dl_4, df_sqrt_dl_5]
df_sqrt_bin_eng = pd.concat(frames_eng)
df_sqrt_bin_dt = pd.concat(frames_dt)
df_sqrt_bin_dl = pd.concat(frames_dl)

df_sqrt_bin = df_sqrt_bin_eng
df_sqrt_bin = df_sqrt_bin.join(df_sqrt_bin_dt)
df_sqrt_bin = df_sqrt_bin.join(df_sqrt_bin_dl)

df_sqrt_bin.fillna(0, inplace=True)
print('df_sqrt_bin:', df_sqrt_bin)

# square function on binary
df_sq_bin = pd.DataFrame()
df_sq_eng_3 = pd.DataFrame()
df_sq_eng_4 = pd.DataFrame()
df_sq_eng_5 = pd.DataFrame()
df_sq_dt_3 = pd.DataFrame()
df_sq_dt_4 = pd.DataFrame()
df_sq_dt_5 = pd.DataFrame()
df_sq_dl_3 = pd.DataFrame()
df_sq_dl_4 = pd.DataFrame()
df_sq_dl_5 = pd.DataFrame()
def mapper(name):
    return name + '_sq'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_sq_eng_3[j] = df_new[j].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_sq_eng_4[j] = df_new[j].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_sq_eng_5[j] = df_new[j].apply(lambda x: np.power(x,2))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_sq_dl_3[k] = df_new[k].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_sq_dl_4[k] = df_new[k].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_sq_dl_5[k] = df_new[k].apply(lambda x: np.power(x,2))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_sq_dt_3[l] = df_new[l].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_sq_dt_4[l] = df_new[l].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_sq_dt_5[l] = df_new[l].apply(lambda x: np.power(x,2))


df_sq_eng_3.rename(mapper=mapper, axis=1, inplace=True)
df_sq_eng_4.rename(mapper=mapper, axis=1, inplace=True)
df_sq_eng_5.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dt_3.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dt_4.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dt_5.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dl_3.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dl_4.rename(mapper=mapper, axis=1, inplace=True)
df_sq_dl_5.rename(mapper=mapper, axis=1, inplace=True)

frames_eng = [df_sq_eng_3, df_sq_eng_4, df_sq_eng_5,]
frames_dt = [df_sq_dt_3, df_sq_dt_4, df_sq_dt_5]
frames_dl = [df_sq_dl_3, df_sq_dl_4, df_sq_dl_5]
df_sq_bin_eng = pd.concat(frames_eng)
df_sq_bin_dt = pd.concat(frames_dt)
df_sq_bin_dl = pd.concat(frames_dl)

df_sq_bin = df_sq_bin_eng
df_sq_bin = df_sq_bin.join(df_sq_bin_dt)
df_sq_bin = df_sq_bin.join(df_sq_bin_dl)

df_sq_bin.fillna(0, inplace=True)
print('df_sq_bin:', df_sq_bin)


# cube function on binary
df_cu_bin = pd.DataFrame()
df_cu_eng_3 = pd.DataFrame()
df_cu_eng_4 = pd.DataFrame()
df_cu_eng_5 = pd.DataFrame()
df_cu_dt_3 = pd.DataFrame()
df_cu_dt_4 = pd.DataFrame()
df_cu_dt_5 = pd.DataFrame()
df_cu_dl_3 = pd.DataFrame()
df_cu_dl_4 = pd.DataFrame()
df_cu_dl_5 = pd.DataFrame()
def mapper(name):
    return name + '_cu'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_cu_eng_3[j] = df_new[j].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_cu_eng_4[j] = df_new[j].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_cu_eng_5[j] = df_new[j].apply(lambda x: np.power(x,3))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_cu_dl_3[k] = df_new[k].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_cu_dl_4[k] = df_new[k].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_cu_dl_5[k] = df_new[k].apply(lambda x: np.power(x,3))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_cu_dt_3[l] = df_new[l].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_cu_dt_4[l] = df_new[l].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_cu_dt_5[l] = df_new[l].apply(lambda x: np.power(x,3))


df_cu_eng_3.rename(mapper=mapper, axis=1, inplace=True)
df_cu_eng_4.rename(mapper=mapper, axis=1, inplace=True)
df_cu_eng_5.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dt_3.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dt_4.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dt_5.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dl_3.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dl_4.rename(mapper=mapper, axis=1, inplace=True)
df_cu_dl_5.rename(mapper=mapper, axis=1, inplace=True)

frames_eng = [df_cu_eng_3, df_cu_eng_4, df_cu_eng_5,]
frames_dt = [df_cu_dt_3, df_cu_dt_4, df_cu_dt_5]
frames_dl = [df_cu_dl_3, df_cu_dl_4, df_cu_dl_5]
df_cu_bin_eng = pd.concat(frames_eng)
df_cu_bin_dt = pd.concat(frames_dt)
df_cu_bin_dl = pd.concat(frames_dl)

df_cu_bin = df_cu_bin_eng
df_cu_bin = df_cu_bin.join(df_cu_bin_dt)
df_cu_bin = df_cu_bin.join(df_cu_bin_dl)

df_cu_bin.fillna(0, inplace=True)
print('df_cu_bin:', df_cu_bin)


# reciprocal function on binary
df_rp_bin = pd.DataFrame()
df_rp_eng_3 = pd.DataFrame()
df_rp_eng_4 = pd.DataFrame()
df_rp_eng_5 = pd.DataFrame()
df_rp_dt_3 = pd.DataFrame()
df_rp_dt_4 = pd.DataFrame()
df_rp_dt_5 = pd.DataFrame()
df_rp_dl_3 = pd.DataFrame()
df_rp_dl_4 = pd.DataFrame()
df_rp_dl_5 = pd.DataFrame()
def mapper(name):
    return name + '_rp'

for i in range(3,6):
    if i == 3:
       Eng = list(df_e_bin_3)
       dl  = list(df_dim_bin_3)
       dt  = list(df_dis_bin_3)
       df_new = pd.DataFrame()
    elif i == 4:
       Eng = list(df_e_bin_4)
       dl  = list(df_dim_bin_4)
       dt  = list(df_dis_bin_4)
       df_new = pd.DataFrame()
    elif i == 5:
       Eng  = list(df_e_bin_5)
       dl   = list(df_dim_bin_5)
       dt   = list(df_dis_bin_5)
       df_new = pd.DataFrame()

    for j in Eng:
        if i == 3:
            df_new[j] =  df_e_bin_3[j]
            df_rp_eng_3[j] = df_new[j].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            df_rp_eng_4[j] = df_new[j].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            df_rp_eng_5[j] = df_new[j].apply(lambda x: np.float_power(x,-1))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            df_rp_dl_3[k] = df_new[k].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            df_rp_dl_4[k] = df_new[k].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            df_rp_dl_5[k] = df_new[k].apply(lambda x: np.float_power(x,-1))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            df_rp_dt_3[l] = df_new[l].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            df_rp_dt_4[l] = df_new[l].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            df_rp_dt_5[l] = df_new[l].apply(lambda x: np.float_power(x,-1))


df_rp_eng_3.rename(mapper=mapper, axis=1, inplace=True)
df_rp_eng_4.rename(mapper=mapper, axis=1, inplace=True)
df_rp_eng_5.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dt_3.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dt_4.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dt_5.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dl_3.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dl_4.rename(mapper=mapper, axis=1, inplace=True)
df_rp_dl_5.rename(mapper=mapper, axis=1, inplace=True)

frames_eng = [df_rp_eng_3, df_rp_eng_4, df_rp_eng_5,]
frames_dt = [df_rp_dt_3, df_rp_dt_4, df_rp_dt_5]
frames_dl = [df_rp_dl_3, df_rp_dl_4, df_rp_dl_5]
df_rp_bin_eng = pd.concat(frames_eng)
df_rp_bin_dt = pd.concat(frames_dt)
df_rp_bin_dl = pd.concat(frames_dl)

df_rp_bin = df_rp_bin_eng
df_rp_bin = df_rp_bin.join(df_rp_bin_dt)
df_rp_bin = df_rp_bin.join(df_rp_bin_dl)

df_rp_bin.fillna(0, inplace=True)
print('df_rp_bin:', df_rp_bin)

"""
# cross grouping on unary(binary)
exp = list(df_exp_bin)
log = list(df_log_bin)
sqrt = list(df_sqrt_bin)
sq = list(df_sq_bin)
cu = list(df_cu_bin)
rp = list(df_rp_bin)

df_ex = pd.DataFrame()
df_lg = pd.DataFrame()
df_st = pd.DataFrame()
df_sr = pd.DataFrame()
df_cb = pd.DataFrame()

for i in exp:
    for j in log:
        df_ex[i + "*" + j] = df_exp_bin[i]*df_log_bin[j]
        df_ex[i + "/" + j] = df_exp_bin[i]/df_log_bin[j]
        df_ex[j + "/" + i] = df_log_bin[j]/df_exp_bin[i]
        print(df_ex)

    for k in sqrt:
        df_ex[i + "*" + k] = df_exp_bin[i]*df_sqrt_bin[k]
        df_ex[i + "/" + k] = df_exp_bin[i]/df_sqrt_bin[k]
        df_ex[k + "/" + i] = df_sqrt_bin[k]/df_exp_bin[i]
        print(df_ex)

    for l in sq:
        df_ex[i + "*" + l] = df_exp_bin[i]*df_sq_bin[l]
        df_ex[i + "/" + l] = df_exp_bin[i]/df_sq_bin[l]
        df_ex[l + "/" + i] = df_sq_bin[l]/df_exp_bin[i]
        print(df_ex)
    
    for m in cu:
        df_ex[i + "*" + m] = df_exp_bin[i]*df_cu_bin[m]
        df_ex[i + "/" + m] = df_exp_bin[i]/df_cu_bin[m]
        df_ex[m + "/" + i] = df_cu_bin[m]/df_exp_bin[i]
        print(df_ex)

    for n in rp:
        df_ex[i + "*" + n] = df_exp_bin[i]*df_rp_bin[n]
        df_ex[i + "/" + n] = df_exp_bin[i]/df_rp_bin[n]
        df_ex[n + "/" + i] = df_rp_bin[n]/df_exp_bin[i]
        print(df_ex)

for i in log:
    for k in sqrt:
        df_lg[i + "*" + k] = df_log_bin[i]*df_sqrt_bin[k]
        df_lg[i + "/" + k] = df_log_bin[i]/df_sqrt_bin[k]
        df_lg[k + "/" + i] = df_sqrt_bin[k]/df_log_bin[i]

    for l in sq:
        df_lg[i + "*" + l] = df_log_bin[i]*df_sq_bin[l]
        df_lg[i + "/" + l] = df_log_bin[i]/df_sq_bin[l]
        df_lg[l + "/" + i] = df_sq_bin[l]/df_log_bin[i]

    for m in cu:
        df_lg[i + "*" + m] = df_log_bin[i]*df_cu_bin[m]
        df_lg[i + "/" + m] = df_log_bin[i]/df_cu_bin[m]
        df_lg[m + "/" + i] = df_cu_bin[m]/df_log_bin[i]

    for n in rp:
        df_lg[i + "*" + n] = df_log_bin[i]*df_rp_bin[n]
        df_lg[i + "/" + n] = df_log_bin[i]/df_rp_bin[n]
        df_lg[n + "/" + i] = df_rp_bin[n]/df_log_bin[i]

for i in sqrt:
    for l in sq:
        df_st[i + "*" + l] = df_sqrt_bin[i]*df_sq_bin[l]
        df_st[i + "/" + l] = df_sqrt_bin[i]/df_sq_bin[l]
        df_st[l + "/" + i] = df_sq_bin[l]/df_sqrt_bin[i]

    for m in cu:
        df_st[i + "*" + m] = df_sqrt_bin[i]*df_cu_bin[m]
        df_st[i + "/" + m] = df_sqrt_bin[i]/df_cu_bin[m]
        df_st[m + "/" + i] = df_cu_bin[m]/df_sqrt_bin[i]

    for n in rp:
        df_st[i + "*" + n] = df_sqrt_bin[i]*df_rp_bin[n]
        df_st[i + "/" + n] = df_sqrt_bin[i]/df_rp_bin[n]
        df_st[n + "/" + i] = df_rp_bin[n]/df_sqrt_bin[i]

for i in sq:
    for m in cu:
        df_sr[i + "*" + m] = df_sq_bin[i]*df_cu_bin[m]
        df_sr[i + "/" + m] = df_sq_bin[i]/df_cu_bin[m]
        df_sr[m + "/" + i] = df_cu_bin[m]/df_sq_bin[i]

    for n in rp:
        df_sr[i + "*" + n] = df_sq_bin[i]*df_rp_bin[n]
        df_sr[i + "/" + n] = df_sq_bin[i]/df_rp_bin[n]
        df_sr[n + "/" + i] = df_rp_bin[n]/df_sq_bin[i]

for i in cu:
    for n in rp:
        df_cb[i + "*" + n] = df_cu_bin[i]*df_rp_bin[n]
        df_cb[i + "/" + n] = df_cu_bin[i]/df_rp_bin[n]
        df_cb[n + "/" + i] = df_rp_bin[n]/df_cu_bin[i]


print(df_ex)
"""

#binary on sqrt of binary fn
def bin_sqrt_bin(i, j, k, l, m, n):
   a, b = i[0], i[1]
   x, y = j[0], j[1]
   c, d = k[0], k[1]
   e, f = l[0], l[1]
   p, q = m[0], m[1]
   r, s = n[0], n[1]

   df_bin_sqrt_eng_3[a +  "+" + b]  = df_sqrt_eng_3[a] + df_sqrt_eng_3[b]
   df_bin_sqrt_eng_3[a +  "+" + b + "_abs"]  = df_bin_sqrt_eng_3[a + "+" + b].abs()
   df_bin_sqrt_eng_3[a +  "-" + b]  = df_sqrt_eng_3[a] - df_sqrt_eng_3[b]
   df_bin_sqrt_eng_3[a +  "-" + b + "_abs"]  = df_bin_sqrt_eng_3[a + "-" + b].abs()
   
   df_bin_sqrt_eng_4[x +  "+" + y]  = df_sqrt_eng_4[x] + df_sqrt_eng_4[y]
   df_bin_sqrt_eng_4[x +  "+" + y + "_abs"]  = df_bin_sqrt_eng_4[x + "+" + y].abs()
   df_bin_sqrt_eng_4[x +  "-" + y]  = df_sqrt_eng_4[x] - df_sqrt_eng_4[y]
   df_bin_sqrt_eng_4[x +  "-" + y + "_abs"]  = df_bin_sqrt_eng_4[x + "-" + y].abs()

   df_bin_sqrt_eng_5[c +  "+" + d]  = df_sqrt_eng_5[c] + df_sqrt_eng_5[d]
   df_bin_sqrt_eng_5[c +  "+" + d + "_abs"]  = df_bin_sqrt_eng_5[c + "+" + d].abs()
   df_bin_sqrt_eng_5[c +  "-" + d]  = df_sqrt_eng_5[c] - df_sqrt_eng_5[d]
   df_bin_sqrt_eng_5[c +  "-" + d + "_abs"]  = df_bin_sqrt_eng_5[c + "-" + d].abs()

   df_bin_sqrt_dt_3[e +  "+" + f]  = df_sqrt_dt_3[e] + df_sqrt_dt_3[f]
   df_bin_sqrt_dt_3[e +  "+" + f + "_abs"]  = df_bin_sqrt_dt_3[e + "+" + f].abs()
   df_bin_sqrt_dt_3[e +  "-" + f]  = df_sqrt_dt_3[e] - df_sqrt_dt_3[f]
   df_bin_sqrt_dt_3[e +  "-" + f + "_abs"]  = df_bin_sqrt_dt_3[e + "-" + f].abs()

   df_bin_sqrt_dt_4[p +  "+" + q]  = df_sqrt_dt_4[p] + df_sqrt_dt_4[q]
   df_bin_sqrt_dt_4[p +  "+" + q + "_abs"]  = df_bin_sqrt_dt_4[p + "+" + q].abs()
   df_bin_sqrt_dt_4[p +  "-" + q]  = df_sqrt_dt_4[p] - df_sqrt_dt_4[q]
   df_bin_sqrt_dt_4[p +  "-" + q + "_abs"]  = df_bin_sqrt_dt_4[p + "-" + q].abs()

   df_bin_sqrt_dt_5[r +  "+" + s]  = df_sqrt_dt_5[r] + df_sqrt_dt_5[s]
   df_bin_sqrt_dt_5[r +  "+" + s + "_abs"]  = df_bin_sqrt_dt_5[r + "+" + s].abs()
   df_bin_sqrt_dt_5[r +  "-" + s]  = df_sqrt_dt_5[r] - df_sqrt_dt_5[s]
   df_bin_sqrt_dt_5[r +  "-" + s + "_abs"]  = df_bin_sqrt_dt_5[r + "-" + s].abs()

   return df_bin_sqrt_eng_3, df_bin_sqrt_eng_4, df_bin_sqrt_eng_5, df_bin_sqrt_dt_3, df_bin_sqrt_dt_4, df_bin_sqrt_dt_5 ;

import itertools
eng_sqrt_3 = list(df_sqrt_eng_3)
eng_sqrt_4 = list(df_sqrt_eng_4)
eng_sqrt_5 = list(df_sqrt_eng_5)
dt_sqrt_3 = list(df_sqrt_dt_3)
dt_sqrt_4 = list(df_sqrt_dt_4)
dt_sqrt_5 = list(df_sqrt_dt_5)
df_bin_sqrt_eng_3 = pd.DataFrame()
df_bin_sqrt_eng_4 = pd.DataFrame()
df_bin_sqrt_eng_5 = pd.DataFrame()
df_bin_sqrt_dt_3 = pd.DataFrame()
df_bin_sqrt_dt_4 = pd.DataFrame()
df_bin_sqrt_dt_5 = pd.DataFrame()
for i,j,k,l,m,n in zip(itertools.combinations(eng_sqrt_3, 2), itertools.combinations(eng_sqrt_4, 2), itertools.combinations(eng_sqrt_5, 2), itertools.combinations(dt_sqrt_3, 2), itertools.combinations(dt_sqrt_4, 2), itertools.combinations(dt_sqrt_5, 2)):
    bin_sqrt_bin(i,j,k,l,m,n)

print(df_bin_sqrt_eng_3)
print(df_bin_sqrt_eng_4)
print(df_bin_sqrt_dt_3)
print(df_bin_sqrt_dt_4)

def bin_sq_bin(i, j, k, l, m, n):
   a, b = i[0], i[1]
   x, y = j[0], j[1]
   c, d = k[0], k[1]
   e, f = l[0], l[1]
   p, q = m[0], m[1]
   r, s = n[0], n[1]

   df_bin_sq_eng_3[a +  "+" + b]  = df_sq_eng_3[a] + df_sq_eng_3[b]
   df_bin_sq_eng_3[a +  "+" + b + "_abs"]  = df_bin_sq_eng_3[a + "+" + b].abs()
   df_bin_sq_eng_3[a +  "-" + b]  = df_sq_eng_3[a] - df_sq_eng_3[b]
   df_bin_sq_eng_3[a +  "-" + b + "_abs"]  = df_bin_sq_eng_3[a + "-" + b].abs()

   df_bin_sq_eng_4[x +  "+" + y]  = df_sq_eng_4[x] + df_sq_eng_4[y]
   df_bin_sq_eng_4[x +  "+" + y + "_abs"]  = df_bin_sq_eng_4[x + "+" + y].abs()
   df_bin_sq_eng_4[x +  "-" + y]  = df_sq_eng_4[x] - df_sq_eng_4[y]
   df_bin_sq_eng_4[x +  "-" + y + "_abs"]  = df_bin_sq_eng_4[x + "-" + y].abs()

   df_bin_sq_eng_5[c +  "+" + d]  = df_sq_eng_5[c] + df_sq_eng_5[d]
   df_bin_sq_eng_5[c +  "+" + d + "_abs"]  = df_bin_sq_eng_5[c + "+" + d].abs()
   df_bin_sq_eng_5[c +  "-" + d]  = df_sq_eng_5[c] - df_sq_eng_5[d]
   df_bin_sq_eng_5[c +  "-" + d + "_abs"]  = df_bin_sq_eng_5[c + "-" + d].abs()

   df_bin_sq_dt_3[e +  "+" + f]  = df_sq_dt_3[e] + df_sq_dt_3[f]
   df_bin_sq_dt_3[e +  "+" + f + "_abs"]  = df_bin_sq_dt_3[e + "+" + f].abs()
   df_bin_sq_dt_3[e +  "-" + f]  = df_sq_dt_3[e] - df_sq_dt_3[f]
   df_bin_sq_dt_3[e +  "-" + f + "_abs"]  = df_bin_sq_dt_3[e + "-" + f].abs()

   df_bin_sq_dt_4[p +  "+" + q]  = df_sq_dt_4[p] + df_sq_dt_4[q]
   df_bin_sq_dt_4[p +  "+" + q + "_abs"]  = df_bin_sq_dt_4[p + "+" + q].abs()
   df_bin_sq_dt_4[p +  "-" + q]  = df_sq_dt_4[p] - df_sq_dt_4[q]
   df_bin_sq_dt_4[p +  "-" + q + "_abs"]  = df_bin_sq_dt_4[p + "-" + q].abs()

   df_bin_sq_dt_5[r +  "+" + s]  = df_sq_dt_5[r] + df_sq_dt_5[s]
   df_bin_sq_dt_5[r +  "+" + s + "_abs"]  = df_bin_sq_dt_5[r + "+" + s].abs()
   df_bin_sq_dt_5[r +  "-" + s]  = df_sq_dt_5[r] - df_sq_dt_5[s]
   df_bin_sq_dt_5[r +  "-" + s + "_abs"]  = df_bin_sq_dt_5[r + "-" + s].abs()

   return df_bin_sq_eng_3, df_bin_sq_eng_4, df_bin_sq_eng_5, df_bin_sq_dt_3, df_bin_sq_dt_4, df_bin_sq_dt_5 ;

import itertools
eng_sq_3 = list(df_sq_eng_3)
eng_sq_4 = list(df_sq_eng_4)
eng_sq_5 = list(df_sq_eng_5)
dt_sq_3 = list(df_sq_dt_3)
dt_sq_4 = list(df_sq_dt_4)
dt_sq_5 = list(df_sq_dt_5)
df_bin_sq_eng_3 = pd.DataFrame()
df_bin_sq_eng_4 = pd.DataFrame()
df_bin_sq_eng_5 = pd.DataFrame()
df_bin_sq_dt_3 = pd.DataFrame()
df_bin_sq_dt_4 = pd.DataFrame()
df_bin_sq_dt_5 = pd.DataFrame()
for i,j,k,l,m,n in zip(itertools.combinations(eng_sq_3, 2), itertools.combinations(eng_sq_4, 2), itertools.combinations(eng_sq_5, 2), itertools.combinations(dt_sq_3, 2), itertools.combinations(dt_sq_4, 2), itertools.combinations(dt_sq_5, 2)):
    bin_sq_bin(i,j,k,l,m,n)

print(df_bin_sq_eng_3)
print(df_bin_sq_eng_4)
print(df_bin_sq_dt_3)
print(df_bin_sq_dt_4)

def bin_cu_bin(i, j, k, l, m, n):
   a, b = i[0], i[1]
   x, y = j[0], j[1]
   c, d = k[0], k[1]
   e, f = l[0], l[1]
   p, q = m[0], m[1]
   r, s = n[0], n[1]

   df_bin_cu_eng_3[a +  "+" + b]  = df_cu_eng_3[a] + df_cu_eng_3[b]
   df_bin_cu_eng_3[a +  "+" + b + "_abs"]  = df_bin_cu_eng_3[a + "+" + b].abs()
   df_bin_cu_eng_3[a +  "-" + b]  = df_cu_eng_3[a] - df_cu_eng_3[b]
   df_bin_cu_eng_3[a +  "-" + b + "_abs"]  = df_bin_cu_eng_3[a + "-" + b].abs()

   df_bin_cu_eng_4[x +  "+" + y]  = df_cu_eng_4[x] + df_cu_eng_4[y]
   df_bin_cu_eng_4[x +  "+" + y + "_abs"]  = df_bin_cu_eng_4[x + "+" + y].abs()
   df_bin_cu_eng_4[x +  "-" + y]  = df_cu_eng_4[x] - df_cu_eng_4[y]
   df_bin_cu_eng_4[x +  "-" + y + "_abs"]  = df_bin_cu_eng_4[x + "-" + y].abs()

   df_bin_cu_eng_5[c +  "+" + d]  = df_cu_eng_5[c] + df_cu_eng_5[d]
   df_bin_cu_eng_5[c +  "+" + d + "_abs"]  = df_bin_cu_eng_5[c + "+" + d].abs()
   df_bin_cu_eng_5[c +  "-" + d]  = df_cu_eng_5[c] - df_cu_eng_5[d]
   df_bin_cu_eng_5[c +  "-" + d + "_abs"]  = df_bin_cu_eng_5[c + "-" + d].abs()

   df_bin_cu_dt_3[e +  "+" + f]  = df_cu_dt_3[e] + df_cu_dt_3[f]
   df_bin_cu_dt_3[e +  "+" + f + "_abs"]  = df_bin_cu_dt_3[e + "+" + f].abs()
   df_bin_cu_dt_3[e +  "-" + f]  = df_cu_dt_3[e] - df_cu_dt_3[f]
   df_bin_cu_dt_3[e +  "-" + f + "_abs"]  = df_bin_cu_dt_3[e + "-" + f].abs()

   df_bin_cu_dt_4[p +  "+" + q]  = df_cu_dt_4[p] + df_cu_dt_4[q]
   df_bin_cu_dt_4[p +  "+" + q + "_abs"]  = df_bin_cu_dt_4[p + "+" + q].abs()
   df_bin_cu_dt_4[p +  "-" + q]  = df_cu_dt_4[p] - df_cu_dt_4[q]
   df_bin_cu_dt_4[p +  "-" + q + "_abs"]  = df_bin_cu_dt_4[p + "-" + q].abs()

   df_bin_cu_dt_5[r +  "+" + s]  = df_cu_dt_5[r] + df_cu_dt_5[s]
   df_bin_cu_dt_5[r +  "+" + s + "_abs"]  = df_bin_cu_dt_5[r + "+" + s].abs()
   df_bin_cu_dt_5[r +  "-" + s]  = df_cu_dt_5[r] - df_cu_dt_5[s]
   df_bin_cu_dt_5[r +  "-" + s + "_abs"]  = df_bin_cu_dt_5[r + "-" + s].abs()

   return df_bin_cu_eng_3, df_bin_cu_eng_4, df_bin_cu_eng_5, df_bin_cu_dt_3, df_bin_cu_dt_4, df_bin_cu_dt_5 ;

import itertools
eng_cu_3 = list(df_cu_eng_3)
eng_cu_4 = list(df_cu_eng_4)
eng_cu_5 = list(df_cu_eng_5)
dt_cu_3 = list(df_cu_dt_3)
dt_cu_4 = list(df_cu_dt_4)
dt_cu_5 = list(df_cu_dt_5)
df_bin_cu_eng_3 = pd.DataFrame()
df_bin_cu_eng_4 = pd.DataFrame()
df_bin_cu_eng_5 = pd.DataFrame()
df_bin_cu_dt_3 = pd.DataFrame()
df_bin_cu_dt_4 = pd.DataFrame()
df_bin_cu_dt_5 = pd.DataFrame()
for i,j,k,l,m,n in zip(itertools.combinations(eng_cu_3, 2), itertools.combinations(eng_cu_4, 2), itertools.combinations(eng_cu_5, 2), itertools.combinations(dt_cu_3, 2), itertools.combinations(dt_cu_4, 2), itertools.combinations(dt_cu_5, 2)):
    bin_cu_bin(i,j,k,l,m,n)

print(df_bin_cu_eng_3)
print(df_bin_cu_eng_4)
print(df_bin_cu_dt_3)
print(df_bin_cu_dt_4)

def bin_rp_bin(i, j, k, l, m, n):
   a, b = i[0], i[1]
   x, y = j[0], j[1]
   c, d = k[0], k[1]
   e, f = l[0], l[1]
   p, q = m[0], m[1]
   r, s = n[0], n[1]

   df_bin_rp_eng_3[a +  "+" + b]  = df_rp_eng_3[a] + df_rp_eng_3[b]
   df_bin_rp_eng_3[a +  "+" + b + "_abs"]  = df_bin_rp_eng_3[a + "+" + b].abs()
   df_bin_rp_eng_3[a +  "-" + b]  = df_rp_eng_3[a] - df_rp_eng_3[b]
   df_bin_rp_eng_3[a +  "-" + b + "_abs"]  = df_bin_rp_eng_3[a + "-" + b].abs()

   df_bin_rp_eng_4[x +  "+" + y]  = df_rp_eng_4[x] + df_rp_eng_4[y]
   df_bin_rp_eng_4[x +  "+" + y + "_abs"]  = df_bin_rp_eng_4[x + "+" + y].abs()
   df_bin_rp_eng_4[x +  "-" + y]  = df_rp_eng_4[x] - df_rp_eng_4[y]
   df_bin_rp_eng_4[x +  "-" + y + "_abs"]  = df_bin_rp_eng_4[x + "-" + y].abs()

   df_bin_rp_eng_5[c +  "+" + d]  = df_rp_eng_5[c] + df_rp_eng_5[d]
   df_bin_rp_eng_5[c +  "+" + d + "_abs"]  = df_bin_rp_eng_5[c + "+" + d].abs()
   df_bin_rp_eng_5[c +  "-" + d]  = df_rp_eng_5[c] - df_rp_eng_5[d]
   df_bin_rp_eng_5[c +  "-" + d + "_abs"]  = df_bin_rp_eng_5[c + "-" + d].abs()

   df_bin_rp_dt_3[e +  "+" + f]  = df_rp_dt_3[e] + df_rp_dt_3[f]
   df_bin_rp_dt_3[e +  "+" + f + "_abs"]  = df_bin_rp_dt_3[e + "+" + f].abs()
   df_bin_rp_dt_3[e +  "-" + f]  = df_rp_dt_3[e] - df_rp_dt_3[f]
   df_bin_rp_dt_3[e +  "-" + f + "_abs"]  = df_bin_rp_dt_3[e + "-" + f].abs()

   df_bin_rp_dt_4[p +  "+" + q]  = df_rp_dt_4[p] + df_rp_dt_4[q]
   df_bin_rp_dt_4[p +  "+" + q + "_abs"]  = df_bin_rp_dt_4[p + "+" + q].abs()
   df_bin_rp_dt_4[p +  "-" + q]  = df_rp_dt_4[p] - df_rp_dt_4[q]
   df_bin_rp_dt_4[p +  "-" + q + "_abs"]  = df_bin_rp_dt_4[p + "-" + q].abs()

   df_bin_rp_dt_5[r +  "+" + s]  = df_rp_dt_5[r] + df_rp_dt_5[s]
   df_bin_rp_dt_5[r +  "+" + s + "_abs"]  = df_bin_rp_dt_5[r + "+" + s].abs()
   df_bin_rp_dt_5[r +  "-" + s]  = df_rp_dt_5[r] - df_rp_dt_5[s]
   df_bin_rp_dt_5[r +  "-" + s + "_abs"]  = df_bin_rp_dt_5[r + "-" + s].abs()

   return df_bin_rp_eng_3, df_bin_rp_eng_4, df_bin_rp_eng_5, df_bin_rp_dt_3, df_bin_rp_dt_4, df_bin_rp_dt_5 ;

import itertools
eng_rp_3 = list(df_rp_eng_3)
eng_rp_4 = list(df_rp_eng_4)
eng_rp_5 = list(df_rp_eng_5)
dt_rp_3 = list(df_rp_dt_3)
dt_rp_4 = list(df_rp_dt_4)
dt_rp_5 = list(df_rp_dt_5)
df_bin_rp_eng_3 = pd.DataFrame()
df_bin_rp_eng_4 = pd.DataFrame()
df_bin_rp_eng_5 = pd.DataFrame()
df_bin_rp_dt_3 = pd.DataFrame()
df_bin_rp_dt_4 = pd.DataFrame()
df_bin_rp_dt_5 = pd.DataFrame()
for i,j,k,l,m,n in zip(itertools.combinations(eng_rp_3, 2), itertools.combinations(eng_rp_4, 2), itertools.combinations(eng_rp_5, 2), itertools.combinations(dt_rp_3, 2), itertools.combinations(dt_rp_4, 2), itertools.combinations(dt_rp_5, 2)):
    bin_rp_bin(i,j,k,l,m,n)

print(df_bin_rp_eng_3)
print(df_bin_rp_eng_4)
print(df_bin_rp_dt_3)
print(df_bin_rp_dt_4)

df_bin_sqrt_eng = df_bin_sqrt_eng_3.append(df_bin_sqrt_eng_4, sort = True)
df_bin_sqrt_eng = df_bin_sqrt_eng.append(df_bin_sqrt_eng_5, sort = True)
df_bin_sqrt_eng.fillna(0, inplace=True)
print(df_bin_sqrt_eng)

df_bin_sqrt_dt = df_bin_sqrt_dt_3.append(df_bin_sqrt_dt_4, sort = True)
df_bin_sqrt_dt = df_bin_sqrt_dt.append(df_bin_sqrt_dt_5, sort = True)
df_bin_sqrt_dt.fillna(0, inplace=True)
print(df_bin_sqrt_dt)

df_bin_sqrt = df_bin_sqrt_eng.join(df_bin_sqrt_dt)
print(df_bin_sqrt)

df_bin_sq_eng = df_bin_sq_eng_3.append(df_bin_sq_eng_4, sort = True)
df_bin_sq_eng = df_bin_sq_eng.append(df_bin_sq_eng_5, sort = True)
df_bin_sq_eng.fillna(0, inplace=True)
print(df_bin_sq_eng)

df_bin_sq_dt = df_bin_sq_dt_3.append(df_bin_sq_dt_4, sort = True)
df_bin_sq_dt = df_bin_sq_dt.append(df_bin_sq_dt_5, sort = True)
df_bin_sq_dt.fillna(0, inplace=True)
print(df_bin_sq_dt)

df_bin_sq = df_bin_sq_eng.join(df_bin_sq_dt)
print(df_bin_sq)

df_bin_cu_eng = df_bin_cu_eng_3.append(df_bin_cu_eng_4, sort = True)
df_bin_cu_eng = df_bin_cu_eng.append(df_bin_cu_eng_5, sort = True)
df_bin_cu_eng.fillna(0, inplace=True)
print(df_bin_cu_eng)

df_bin_cu_dt = df_bin_cu_dt_3.append(df_bin_cu_dt_4, sort = True)
df_bin_cu_dt = df_bin_cu_dt.append(df_bin_cu_dt_5, sort = True)
df_bin_cu_dt.fillna(0, inplace=True)
print(df_bin_cu_dt)

df_bin_cu = df_bin_cu_eng.join(df_bin_cu_dt)
print(df_bin_cu)

df_bin_rp_eng = df_bin_rp_eng_3.append(df_bin_rp_eng_4, sort = True)
df_bin_rp_eng = df_bin_rp_eng.append(df_bin_rp_eng_5, sort = True)
df_bin_rp_eng.fillna(0, inplace=True)
print(df_bin_rp_eng)

df_bin_rp_dt = df_bin_rp_dt_3.append(df_bin_rp_dt_4, sort = True)
df_bin_rp_dt = df_bin_rp_dt.append(df_bin_rp_dt_5, sort = True)
df_bin_rp_dt.fillna(0, inplace=True)
print(df_bin_rp_dt)

df_bin_rp = df_bin_rp_eng.join(df_bin_rp_dt)
print(df_bin_rp)

"""
#cross grouping on binary on unary(binary)
st = list(df_bin_sqrt)
sr = list(df_bin_sq)
cb = list(df_bin_cu)
rs = list(df_bin_rp)

df_b_st = pd.DataFrame()
df_b_sr = pd.DataFrame()
df_b_cb = pd.DataFrame()
df_b_rs = pd.DataFrame()

for i in st:
    for j in sr:
        df_b_st[i + "*" + j] = df_bin_sqrt[i]*df_bin_sq[j]
        df_b_st[i + "/" + j] = df_bin_sqrt[i]/df_bin_sq[j]
        df_b_st[j + "/" + i] = df_bin_sq[j]/df_bin_sqrt[i]

    for m in cb:
        df_b_st[i + "*" + m] = df_bin_sqrt[i]*df_bin_cu[m]
        df_b_st[i + "/" + m] = df_bin_sqrt[i]/df_bin_cu[m]
        df_b_st[m + "/" + i] = df_bin_cu[m]/df_bin_sqrt[i]

    for n in rs:
        df_b_st[i + "*" + n] = df_bin_sqrt[i]*df_bin_rp[n]
        df_b_st[i + "/" + n] = df_bin_sqrt[i]/df_bin_rp[n]
        df_b_st[n + "/" + i] = df_bin_rp[n]/df_bin_sqrt[i]

for i in sr:
    for m in cb:
        df_b_sr[i + "*" + m] = df_bin_sq[i]*df_bin_cu[m]
        df_b_sr[i + "/" + m] = df_bin_sq[i]/df_bin_cu[m]
        df_b_sr[m + "/" + i] = df_bin_cu[m]/df_bin_sq[i]

    for n in rs:
        df_b_sr[i + "*" + n] = df_bin_sq[i]*df_bin_rp[n]
        df_b_sr[i + "/" + n] = df_bin_sq[i]/df_bin_rp[n]
        df_b_sr[n + "/" + i] = df_bin_rp[n]/df_bin_sq[i]

for i in cb:
    for n in rs:
        df_b_cb[i + "*" + n] = df_bin_cu[i]*df_bin_rp[n]
        df_b_cb[i + "/" + n] = df_bin_cu[i]/df_bin_rp[n]
        df_b_cb[n + "/" + i] = df_bin_rp[n]/df_bin_cu[i]

print(df_b_st)
"""

df_e_bin = df_e_bin_3.append(df_e_bin_4, sort=True)
df_e_bin = df_e_bin.append(df_e_bin_5, sort=True)
df_e_bin.fillna(0, inplace=True)
print(df_e_bin)

df_dim_bin = df_dim_bin_3.append(df_dim_bin_4, sort=True)
df_dim_bin = df_dim_bin.append(df_dim_bin_5, sort=True)
df_dim_bin.fillna(0, inplace=True)
print(df_dim_bin)

df_dis_bin = df_dis_bin_3.append(df_dis_bin_4, sort=True)
df_dis_bin = df_dis_bin.append(df_dis_bin_5, sort=True)
df_dis_bin.fillna(0, inplace=True)
print(df_dis_bin)

# join dataframes of df and all unary
df = df.join(df_exp)
df = df.join(df_log)
df = df.join(df_sqrt)
df = df.join(df_sq)
df = df.join(df_cu)
df = df.join(df_rp)
df = df.join(df_e_bin)
df = df.join(df_dim_bin)
df = df.join(df_dis_bin)
df = df.join(df_exp_bin)
df = df.join(df_log_bin)
df = df.join(df_sqrt_bin)
df = df.join(df_sq_bin)
df = df.join(df_cu_bin)
df = df.join(df_rp_bin)
df = df.join(df_bin_sqrt)
df = df.join(df_bin_sq)
df = df.join(df_bin_cu)
df = df.join(df_bin_rp)


print('df:', '\n', df)

df.fillna(df.mean(), inplace=True)
print('df:','\n', df)

# shuffling the rows of dataframe to remove bias
from sklearn.utils import shuffle
df = shuffle(df, random_state =2)
df = df.reset_index(drop=True)
print('shuffled_df:','\n',df)

df_feat = df.iloc[:,11:]
print('features_df:','\n',df_feat)
cols = df_feat.columns.values
print(cols)
cols = shuffle(df_feat.columns.values, random_state = 2)
print(cols)
df_feat_shuffled = df_feat[cols]
print(df_feat_shuffled)
X=df_feat_shuffled.iloc[:,0:].values
print('inf_values:', np.isinf(X).any())
np.nan_to_num(X, copy=False)
print('inf_values:', np.isinf(X).any())
print('X:',X)
df_cohes = df.iloc[:,10]
print('df_cohes:', '\n', df_cohes)
y=df.iloc[:,10].values
print('Y:',y)

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)
print('X_Scaled:',X_Scaled)
print(np.isnan(X_Scaled).any(),np.isinf(X_Scaled).any())
np.nan_to_num(X_Scaled, copy=False)
print(np.isnan(X_Scaled).any(),np.isinf(X_Scaled).any())

from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(test_size=0.3, random_state=2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X_Scaled,y, test_size=0.3, random_state=2) 
print(X_train, '\n', y_train)
print(X_train.shape, X_test.shape)


c = 0
d = 0
alp_1 = []
mse_lasso = []
mse_svm = []
for j in range(0,100):
    d = d + 0.001
    b = float("{0:.3f}".format(d))
    print(b)
    alp_1.append(b)

""" 
    #svr_rbf = SVR(kernel='rbf', gamma=b)
    #parameters = {'C':[1e3]}
    #clf = GridSearchCV(estimator=svr_rbf, param_grid=parameters, cv=cv)
    #clf.fit(X_Scaled, y)
    #y_pred_svm = clf.predict(X)
    #mse_svm.append(mean_squared_error(y, y_pred_svm))
"""

#applying lasso regression with CV
l2_model = LassoCV(n_alphas=50, alphas=alp_1, cv=cv, fit_intercept= True)                 # alpha=0 same as linear regression
#l2_model.fit(X_Scaled, y)
#y_pred_lasso = l2_model.predict(X_Scaled)
#print("y:", y)
#print("y_pred_lasso:", y_pred_lasso)
#print('MSE_Lasso:', mean_squared_error(y, y_pred_lasso))

"""
rfe_err = []
feat = []
# applying recursive feature elimination on LassoCV
for b in range(1,140,7):
  feat.append(b)
  selector = RFE(l2_model, n_features_to_select=b, step=0.5)
  selector.fit(X_Scaled, y)
  y_pred_RFE_lasso = selector.predict(X_Scaled)
  print("y_pred_RFE:", y_pred_RFE_lasso)
   #cols = selector.get_support()
   #new_features = df_feat.columns[cols]
   #print(new_features)
  error = mean_squared_error(y,y_pred_RFE_lasso) 
  print("MSE_RFE_lasso:", error)
  rfe_err.append(error)

print(rfe_err)
print(feat)

rfe_df = pd.DataFrame({'feat':feat})
rfe_df['rfe_error'] = rfe_err
rfe_df.to_excel("rfe_error.xlsx")
print(rfe_err)
"""

#applying RFE with optimal number of features
selector = RFE(l2_model, n_features_to_select=85, step=0.5)
selector.fit(X_Scaled, y)
y_pred_RFE_lasso = selector.predict(X_Scaled)
print("y_pred_RFE:", y_pred_RFE_lasso)
cols = selector.get_support()
print(cols)
new_features = df_feat_shuffled.columns[cols]
new_optimal_df_feat = pd.DataFrame()
new_optimal_df_feat = df_feat_shuffled[new_features]
print(new_features)
print(new_optimal_df_feat)

for i in range(0,85):
    print(new_features[i])

#scatter_plot_with_correlation_line(y, y_pred_lasso, 'lassoCV_newfeat_newdat_cohes_total.png')
