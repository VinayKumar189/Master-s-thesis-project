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


def scatter_plot_with_correlation_line(x, y, graph_filepath):
    plt.scatter(x, y)
    axes = plt.gca()
    m, b = np.polyfit(x, y, 1)
    X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
    plt.xlabel("Calculated electronic bandgap (eV)")
    plt.ylabel("Predicted electronic bandgap (eV)")
    plt.plot(X_plot, m*X_plot + b, '-')
    plt.savefig(graph_filepath, dpi=1000, format='png', bbox_inches='tight')
    plt.show()

df = pd.read_csv('newfeatures_data.csv', sep=',')
print('df:','\n',df)

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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.exp(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.exp(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.exp(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.exp(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.log(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.log(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.log(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.log(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.log(x))

    print('df_new:', df_new)
    df_log_bin = df_log_bin.append(df_new)

df_log_bin.rename(mapper=mapper, axis=1, inplace=True)
df_log_bin.fillna(0, inplace=True)
print('df_log_bin:', df_log_bin)


# square root function on binary
df_sqrt_bin = pd.DataFrame()
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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.sqrt(x))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.sqrt(x))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.sqrt(x))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.sqrt(x))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.sqrt(x))

    print('df_new:', df_new)
    df_sqrt_bin = df_sqrt_bin.append(df_new)

df_sqrt_bin.rename(mapper=mapper, axis=1, inplace=True)
df_sqrt_bin.fillna(0, inplace=True)
print('df_sqrt_bin:', df_sqrt_bin)


# square function on binary
df_sq_bin = pd.DataFrame()
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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,2))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,2))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,2))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,2))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,2))

    print('df_new:', df_new)
    df_sq_bin = df_sq_bin.append(df_new)

df_sq_bin.rename(mapper=mapper, axis=1, inplace=True)
df_sq_bin.fillna(0, inplace=True)
print('df_sq_bin:', df_sq_bin)


# cube function on binary
df_cu_bin = pd.DataFrame()
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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.power(x,3))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.power(x,3))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,3))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,3))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.power(x,3))

    print('df_new:', df_new)
    df_cu_bin = df_cu_bin.append(df_new)

df_cu_bin.rename(mapper=mapper, axis=1, inplace=True)
df_cu_bin.fillna(0, inplace=True)
print('df_cu_bin:', df_cu_bin)


# reciprocal function on binary
df_rp_bin = pd.DataFrame()
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
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[j] =  df_e_bin_4[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[j] =  df_e_bin_5[j]
            print(df_new[j])
            df_new[j] = df_new[j].apply(lambda x: np.float_power(x,-1))

    for k in dl:
        if i == 3:
            df_new[k] =  df_dim_bin_3[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[k] =  df_dim_bin_4[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[k] =  df_dim_bin_5[k]
            print(df_new[k])
            df_new[k] = df_new[k].apply(lambda x: np.float_power(x,-1))

    for l in dt:
        if i == 3:
            df_new[l] =  df_dis_bin_3[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.float_power(x,-1))
        elif i == 4:
            df_new[l] =  df_dis_bin_4[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.float_power(x,-1))
        elif i == 5:
            df_new[l] =  df_dis_bin_5[l]
            print(df_new[l])
            df_new[l] = df_new[l].apply(lambda x: np.float_power(x,-1))

    print('df_new:', df_new)
    df_rp_bin = df_rp_bin.append(df_new)

df_rp_bin.rename(mapper=mapper, axis=1, inplace=True)
df_rp_bin.fillna(0, inplace=True)
print('df_rp_bin:', df_rp_bin)

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


# join dataframes of all
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

print('df:', '\n', df)

df.fillna(df.mean(), inplace=True)
print('df:','\n', df)

# shuffling the rows of dataframe to remove bias
from sklearn.utils import shuffle
df = shuffle(df, random_state =2)
df = df.reset_index(drop=True)
print('shuffled_df:','\n',df)


X=df.iloc[:,11:].values
df_feat = df.iloc[:,11:]
print('features_df:','\n',df_feat)
print('inf_values:', np.isinf(X).any())
np.nan_to_num(X, copy=False)
print('inf_values:', np.isinf(X).any())
print('X:',X)
df_band = df.iloc[:,2]
print('df_band:','\n', df_band)
y=df.iloc[:,2].values
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


from sklearn.model_selection import GridSearchCV
c = 0
d = 0
alp_1 = []
mse_lasso = []
mse_svm = []
for j in range(0,100):
    d = d + 0.001
    d = d + 0.001
    b = float("{0:.3f}".format(d))
    print(b)
    alp_1.append(b)
    
 
    #svr_rbf = SVR(kernel='rbf', gamma=b)
    #parameters = {'C':[1e3]}
    #clf = GridSearchCV(estimator=svr_rbf, param_grid=parameters, cv=cv)
    #clf.fit(X_Scaled, y)
    #y_pred_svm = clf.predict(X)
    #mse_svm.append(mean_squared_error(y, y_pred_svm))

l2_model = LassoCV(n_alphas=50, alphas=alp_1, cv=cv, fit_intercept= True)                 # alpha=0 same as linear regression
#l2_model.fit(X_Scaled, y)
#y_pred_lasso = l2_model.predict(X_Scaled)
#print("y:", y)
#print("y_pred_lasso:", y_pred_lasso)
#print('MSE_Lasso:', mean_squared_error(y, y_pred_lasso))

"""
rfe_error = []
feat = []
# applying RFE on feature set
for b in range(1,200,10):
    feat.append(b)
    selector = RFE(l2_model, n_features_to_select=b, step=0.6)
    selector.fit(X_Scaled,y)
    y_pred_RFE_lasso = selector.predict(X_Scaled)
    print("y_pred_RFE_lasso:", y_pred_RFE_lasso)
    #cols = selector.get_support()
    #new_features = df_feat.columns[cols]
    #print(new_features)
    error = mean_squared_error(y,y_pred_RFE_lasso)
    print("MSE_Lasso_RFE:", error)
    rfe_error.append(error)


rfe_df = pd.DataFrame({'feat':feat})
rfe_df['rfe_error'] = rfe_error
print(rfe_df)
print(feat)
print(rfe_error)
"""
#applying RFE with optimal number of features
selector = RFE(l2_model, n_features_to_select=61, step=0.6)
selector.fit(X_Scaled,y)
y_pred_RFE_lasso = selector.predict(X_Scaled)
print("y_pred_RFE_lasso:", y_pred_RFE_lasso)
cols = selector.get_support()
new_features = df_feat.columns[cols]
print(new_features)

for i in range(1,21):
    print(new_features[i])
    
scatter_plot_with_correlation_line(y, y_pred_RFE_lasso, 'lassoCV_newfeat_olddata_band_rfe.png')
