# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:10:27 2021

@author: teywa 
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, FunctionTransformer
from sklearn.preprocessing import PowerTransformer, Normalizer, OrdinalEncoder, OneHotEncoder, KBinsDiscretizer, Binarizer  
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, XGBRegressor
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, f_classif, chi2, mutual_info_classif, mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="centered", initial_sidebar_state="expanded")

st.write("""
# Data Science WEB APP | KATALYSE IS
""")

global df

def list_polynom(variable, deg):
    gh = []
    for k in range(deg):
        temp = combinations_with_replacement(variable, k+1)
        for i in list(temp):
            gh.append(i)
    return gh

def fc(a):
    boole = False
    if (len(a) > 1):
        i = 0
        while (i < len(a) and boole == False):
            j = i + 1
            while(j < (len(a)) and boole == False):
                if (a[i] == a[j]):
                    boole = True
                j = j + 1
            i = i + 1     
    return boole

def list_polynom_oi(variable, deg):
    gh = list_polynom(variable, deg)
    wi = []
    for i in range(len(gh)):
        if (fc(gh[i]) == False):
            wi.append(gh[i])
    return wi

def obtain_list(wi):
    fin = []
    for i in wi:
        if(len(i) > 1):
            a = ""
            j = 0
            while j < len(i):
                a = a + i[j]
                if (j != len(i) - 1):
                    a = a + "*"
                j = j+1
            fin.append(a)
    return fin

def calcul_pol(l,df):
    lf = []
    for i in range(len(l)):
        temp = l[i].split("*")
        temp2 = df[temp[0]]
        for j in range(1,len(temp)):
            temp2 = temp2*df[temp[j]]
        lf.append(temp2.tolist())
    return np.array(lf)

def affichage_result_classification(y_true,y_pred):
    
    st.write("")
    st.write("Matrice de confusion :")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    st.pyplot()
                                    
    st.write("")
    st.write("Accuracy Score")
    st.text(accuracy_score(y_true, y_pred))
                                    
    st.write("")
    st.write("Roc Auc Score")
    st.text(roc_auc_score(y_true, y_pred))
                                    
    st.write("")
    st.write("Precision Score")
    st.text(precision_score(y_true, y_pred))
                                    
    st.write("")
    st.write("Recall Score")
    st.text(recall_score(y_true, y_pred))
                                    
    st.write("")
    st.write("F1 Score")
    st.text(f1_score(y_true, y_pred)) 

    return  

def affichage_result_regression(y_true,y_pred):
                                 
    st.write("")
    st.write("R2 Score")
    st.text(r2_score(y_true, y_pred))
                                    
    st.write("")
    st.write("MSE")
    st.text(mean_squared_error(y_true, y_pred))
                                    
    st.write("")
    st.write("MAE (mean)")
    st.text(mean_absolute_error(y_true, y_pred))
                                    
    st.write("")
    st.write("MAE (median)")
    st.text(median_absolute_error (y_true, y_pred))

    return 

def affichage_apres_processing(op):
    
    st.write("X_train et y_train")
    st.write(st.session_state.X_train)
    st.write("X_test et y_test")
    st.write(st.session_state.X_test)
    if ("op" != "no"):
        st.session_state.listeop.append(op)
    st.success("Opération enregistrée")
    
    return 

def op_avant_algo(var_cible):
    
    if (st.session_state.retrain == 0):
    
        st.session_state.y_train = st.session_state.X_train[var_cible]
        st.session_state.y_test = st.session_state.X_test[var_cible]
        st.session_state.X_train.drop(var_cible, inplace=True, axis=1)
        st.session_state.X_test.drop(var_cible, inplace=True, axis=1)
        st.session_state.retrain = 1
                                
    return 


def wrapper_method(model,bol,scoring,X_train,y_train):
    sfs = SFS(model,
              k_features=(1,X_train.shape[1]),
              forward=bol,
              floating=False,
              scoring = scoring,
              cv = 2)
    sfs.fit(X_train, y_train)
    return sfs.k_feature_names_

def apply_operation_prepro(scaler, var_choisi):
    st.session_state.X_train[var_choisi] = scaler.fit_transform(st.session_state.X_train[var_choisi])
    st.session_state.X_test[var_choisi] = scaler.transform(st.session_state.X_test[var_choisi])
    return 

def concat_list(LL):
    L = []
    for i in range(len(LL)):
        L = L + LL[i]
    return L

def classer_list_selon_taille(bl):
    nbl = bl.copy()
    nbl.sort(key=len)
    return nbl[::-1]

def create_Pipe_prepro(G, subset, list_variable_in_L, i):
    steps = []
    for h in G:
        steps.append(('pipe_prepro' + str(i+100+h), eval(list_variable_in_L[h].split("/")[0])))
    return steps

def extract_list_var_from_L(L):
    list_variable_in_L = []
    list_str_in_L = []
    for i in range(len(L)):
        a = L[i].split("/")
        if (a[1] == "var"):
            list_variable_in_L.append(a[2].split(","))
            list_str_in_L.append(L[i])
    return list_variable_in_L, list_str_in_L

def find_good_subset(L,order_list_size,list_of_var):
    subset = []
    for i in range(len(order_list_size)):
        j = i + 1
        bol = False
        while (j < len(order_list_size) and (bol == False)):
            if ((set(order_list_size[j]) <= set(order_list_size[i])) == True):
                bol = True   
            j = j + 1
        if (bol == False):
            subset.append(order_list_size[i])
      
    subset_eclat = list(set(concat_list(subset)))
    
    for i in range(len(list_of_var)):
        if (list_of_var[i] not in subset_eclat):
            subset.append([list_of_var[i]])
    
    return subset

def find_operation(L, subset, list_str_in_L, list_var_in_L):
    list_of_action = []
    pipes_prepro = []
    for i in range(len(subset)):
        G = []
        for j in range(len(list_str_in_L)): 
            if(set(subset[i]).issubset(set(list_var_in_L[j]))):
                G.append(j)
        if (len(G) != 0):
            list_of_action.append(create_Pipe_prepro(G, subset, list_str_in_L, i))
    for i in range(len(list_of_action)):
        pipes_prepro.append(('pipe_ct' + str(i),Pipeline(list_of_action[i]),subset[i]))
    return pipes_prepro

def affichage_pipeline(L):

    list_var_in_L, list_str_in_L = extract_list_var_from_L(L)
    list_of_var = list(set(concat_list(list_var_in_L)))
    order_list_size = classer_list_selon_taille(list_var_in_L)
    subset = find_good_subset(L,order_list_size,list_of_var) 
    pipes_prepro = find_operation(L, subset, list_str_in_L, list_var_in_L) 

    if ((L[-1].split('/')[0]) == "XGBClassifier"):
        dictionnaire = L[-1].split('/')[2]
        model =  'passthrough'
        st.text("model = XGBClassifier(" + dictionnaire + ")")
        st.text(Pipeline([('preprocessor',ColumnTransformer(pipes_prepro)), ('model',model)]))
    elif ((L[-1].split('/')[0]) == "XGBRegressor"):
        dictionnaire = L[-1].split('/')[2]
        model =  'passthrough'
        st.text("model = XGBRegressor(" + dictionnaire + ")")
        st.text(Pipeline([('preprocessor',ColumnTransformer(pipes_prepro)), ('model',model)]))
    else: 
        st.text(Pipeline([('preprocessor',ColumnTransformer(pipes_prepro)), ('model',eval(L[-1].split('/')[0]))]))
    return


st.sidebar.write(""" # Configurateur """)

file = st.sidebar.file_uploader(label="Upload un fichier csv, txt ou xlsx", type=['csv', 'xlsx', 'txt'])

activite = ["Exploration des données","Visualisation des données","Construction du modèle"]	
rubrique = st.sidebar.selectbox("Mode",activite)

   
if (file is not None):
    
    file_format = file.name.split('.')[1]
    df = pd.DataFrame()
    if (file_format == "xlsx"):
        df = pd.read_excel(file)
    elif (file_format == "csv"):
        df = pd.read_csv(file,sep=None, engine="python")
    elif (file_format == "txt"):
        df = pd.read_table(file,sep=None, engine="python")

    if (df.shape[0] != 0):
        
        all_columns_names = df.columns.tolist()
        var_cont = st.sidebar.multiselect('Variable continue', all_columns_names)
        var_cat = st.sidebar.multiselect('Variable catégorielle', all_columns_names)
        var_cible = st.sidebar.multiselect('Variable cible', all_columns_names) 
        
        if 'dataframe' not in st.session_state:
            st.session_state.dataframe = df.copy()
        if 'listeop' not in st.session_state:
            st.session_state.listeop = []
        if 'split' not in st.session_state:
            st.session_state.split = False
        if 'X_train' not in st.session_state:
            st.session_state.X_train = pd.DataFrame()
        if 'X_test' not in st.session_state:
            st.session_state.X_test = pd.DataFrame()
        if 'y_train' not in st.session_state:
            st.session_state.y_train = pd.DataFrame()
        if 'y_test' not in st.session_state:
            st.session_state.y_test = pd.DataFrame()
        if 'l1_gp' not in st.session_state:
            st.session_state.l1_gp = []
        if 'l2_gp' not in st.session_state:
            st.session_state.l2_gp = []
        if 'retrain' not in st.session_state:
            st.session_state.retrain = 0

if rubrique == 'Exploration des données':
    st.write(" ")
    st.write(" ")
    st.subheader('Mode exploration des données')
    st.write(" ")
    st.write(" ")
    st.write(""" ##### Voici le dataframe """)
    
    try :
        st.write(" ")
        st.write(df)
    except Exception:
        st.write(" ")
        st.write("Il faut d'abord upload un fichier")
        
    st.write(" ")
    st.write(" ")
    st.write(""" ##### Exploration """)
    st.write(" ") 
    
    if st.checkbox("Taille du dataframe"):
        st.write(df.shape)

    if st.checkbox("Résumé statistique"):
        st.write(df.describe())

    if st.checkbox("Nombre de valeurs distincts d'une colonne"):
        colonne = st.selectbox("Choix de la colonne", df.columns.tolist())
        st.write(df[colonne].value_counts())
        
    if st.checkbox("Valeurs manquantes par colonne"):
        st.text(df.isnull().sum())
        
    if st.checkbox("Type reconnu"):
        st.text(df.dtypes)

    if st.checkbox("Matrice des corrélations"):
        fig = plt.figure()
        sns.heatmap(df.corr(),annot=True, cmap = "coolwarm")
        st.pyplot(fig)     

elif rubrique == 'Visualisation des données':
    
    st.write(" ")
    st.write(" ")
    st.subheader('Mode visualisation des données')
    st.write(" ")
    st.write(" ")
    st.write(""" ##### Voici le dataframe """)
    
    try :
        st.write(" ")
        st.write(df)
    except Exception:
        st.write(" ")
        st.write("Il faut d'abord upload un fichier")
        
    if file is not None:   
        st.write(" ")
        st.write(""" ##### Type de graphique """)
        a = st.radio(" ", ["Graphique d'une variable continue", 
                           "Graphique d'une variable catégorielle",
                           "Graphique de deux variables continues",
                           "Graphique d'une variable continue et une variable catégorielle",
                           "Graphique de toutes les variables continues"])
        st.write(" ")
        all_columns_names = df.columns.tolist() 
        
        if (a == "Graphique d'une variable continue"):
            type_of_plot = st.selectbox("Choix du type de graphique",["Boite à moustache","Violin","Distribution"])
            st.write(" ")
            selected_columns_name = st.selectbox("Choix de la variable",var_cont)
            st.write(" ")
            
            if (type_of_plot == "Distribution"):
                bins = st.slider("Sélectionne le nombre de bins", 2, 15)            
           
            if st.button("Générer le graphique"):
                
                if type_of_plot == 'Boite à moustache':
                    fig = plt.figure()
                    sns.boxplot(data=df, x = selected_columns_name)
                    st.pyplot(fig)
                
                elif type_of_plot == 'Violin':
                    fig = plt.figure()
                    sns.violinplot(data=df, x = selected_columns_name)
                    st.pyplot(fig)
                
                elif type_of_plot == 'Distribution':
                    fig = plt.figure()
                    sns.distplot(df[selected_columns_name], kde = True, bins = bins) 
                    st.pyplot(fig)   
        
        elif (a == "Graphique d'une variable catégorielle"):
             type_of_plot = st.selectbox("Choix du type de graphique",["Distribution"])
             st.write(" ")
             selected_columns_name = st.selectbox("Choix de la variable",var_cat)
             st.write(" ")
             
             if st.button("Générer le graphique"):
                 
                 if type_of_plot == 'Distribution':
                        fig = plt.figure()
                        sns.countplot(x=selected_columns_name, data=df) 
                        st.pyplot(fig)              
        
        elif (a == "Graphique de deux variables continues"):
            type_of_plot = st.selectbox("Choix du type de graphique",["Ligne","Nuage de points"])
            st.write(" ")
            selected_columns_name = st.multiselect("Choix des variables",var_cont)
            st.write(" ")
           
            if (len(selected_columns_name) != 2):
                st.write("Il n'y a pas le bon nombre de variables")
           
            elif st.button("Générer le graphique"):
               
                if (type_of_plot == "Ligne"):
                  sns.relplot(data=df, kind = 'line', x = selected_columns_name[0], y = selected_columns_name[1])
                  st.pyplot()
                
                elif (type_of_plot == "Nuage de points"):
                  sns.relplot(data=df, kind = 'scatter', x = selected_columns_name[0], y = selected_columns_name[1])
                  st.pyplot()     
       
        elif (a == "Graphique d'une variable continue et une variable catégorielle"):
            type_of_plot = st.selectbox("Choix du type de graphique",["Violin","Boite à moustache"])
            selected_columns_name = st.selectbox("Choix de la variable continue",var_cont)
            selected_columns_name2 = st.selectbox("Choix de la variable catégorielle",var_cat)
            
            if st.button("Générer le graphique"):
                
                if (type_of_plot == "Boite à moustache"):
                    sns.catplot(data=df, kind = 'box', x = selected_columns_name2, y = selected_columns_name)
                    st.pyplot()
               
                elif (type_of_plot == "Violin"):
                    sns.catplot(data=df, kind = 'violin', x = selected_columns_name2, y = selected_columns_name)
                    st.pyplot()   
       
        elif (a == "Graphique de toutes les variables continues"):
            type_of_plot = st.selectbox("Choix du type de graphique",["Histogrammes","Graphiques 2 à 2"])
            st.write(" ")
            
            if st.button("Générer le graphique"):
                
                if (type_of_plot == "Graphiques 2 à 2"):
                    sns.pairplot(data = df[var_cont], diag_kind = 'kde')
                    st.pyplot()
               
                elif (type_of_plot == "Histogrammes"):
                    fig = plt.figure()
                    df.hist(figsize=(15,30)) 
                    st.pyplot(fig)
                
elif rubrique == 'Construction du modèle':
    
    st.write(" ")
    st.write(" ")
    st.subheader('Mode construction du modèle')
    st.write(" ")
    st.write(" ")
    st.write(""" ##### Voici le dataframe """)
    
    try :
        
        if (st.session_state.X_train.shape == (0,0)):   
            st.write(" ")
            st.write(st.session_state.dataframe)
       
        else :
            st.write("X_train et y_train")
            st.write(st.session_state.X_train)
            st.write("X_test et y_test")
            st.write(st.session_state.X_test)
            
        if st.button("Rafraichir dataframe"):
            st.success("Sucess")
            
    except Exception:
        st.write(" ")
        st.write("Il faut d'abord upload un fichier")

    if file is not None:
        
        if (st.session_state.split == False):
            
            st.write("Il faut split le dataframe")
            st.write("Affecter l'ensemble des variables souhaitées (continues / catégorielles / cible) ci-contre")
            ron = st.radio("Randomisation", [True,False])
            tdt = st.text_input("Entrer la taille de la base de test", "Réel entre 0 et 1 (Ex : 0.33)")
                
            if st.button("Effectuer l'opération"):
                if ((len(var_cont) != 0) or (len(var_cat) != 0)):
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
                        st.session_state.dataframe[var_cont + var_cat + var_cible], st.session_state.dataframe[var_cible], test_size=float(tdt), shuffle = ron, random_state=42)
                    st.session_state.split = True
                    st.write("X_train et y_train")
                    st.write(st.session_state.X_train)
                    st.write("X_test et y_test")
                    st.write(st.session_state.X_test)
                else:
                    st.write("Affecter l'ensemble des variables souhaitées (continues / catégorielles / cible) ci-contre")   
        
        else :
            
            if st.button("Voir les opérations enregistrées"):
              
                if st.session_state.listeop == []:
                    st.write("Aucune opération effectuée")
              
                else : 
                    st.write(st.session_state.listeop)

            st.write(" ") 
            st.write(""" ##### Effectuer une nouvelle opération """) 
            b = st.selectbox("Sélectionner le type d'opération",["Nettoyage du dataset", "Feature Selection", "Preprocessing","Modèle"])
           
            if (b == "Nettoyage du dataset"):
                mvo = st.radio("Méthode de gestion", ["Supprimer les données contenant des valeurs manquantes","Remplacer les valeurs manquantes",
                                                      "Supprimer les doublons", "Supprimer des colonnes"])
             
                if (mvo == "Supprimer les données contenant des valeurs manquantes"):
                    nd = st.radio("Suppression des colonnes", ["Nettoyer l'ensemble du dataframe", "Nettoyer les lignes à partir de certaines colonnes"])
               
                    if (nd == "Nettoyer l'ensemble du dataframe"):
                  
                        if st.button("Voir et enregistrer l'opération"):
                            st.session_state.X_train.dropna(inplace=True)
                            st.session_state.X_test.dropna(inplace=True)
                            affichage_apres_processing("no/no")
                  
                    elif (nd == "Nettoyer les lignes à partir de certaines colonnes"):
                        var_choisi = st.multiselect("Choix des variables",all_columns_names)
                    
                        if st.button("Voir et enregistrer l'opération"):
                            st.session_state.X_train.dropna(subset=var_choisi, inplace=True)
                            st.session_state.X_test.dropna(subset=var_choisi, inplace=True)
                            affichage_apres_processing("no/no")
                
                elif (mvo == "Remplacer les valeurs manquantes"):
                    type_variable = st.radio("Type de variable", ["Variable continue","Variable catégorielle"])
                 
                    if (type_variable == "Variable continue"): 
                        strat_remp = st.radio("Méthode de remplacement", ["mean","median","most_frequent","constant"])
                        var_choisi = st.multiselect("Choix des variables",var_cont)
                        ftp = "var/" + ','.join(var_choisi) + "/" + str(strat_remp)
                
                    elif (type_variable == "Variable catégorielle"):
                        strat_remp = st.radio("Méthode de remplacement", ["most_frequent","constant"])
                        var_choisi = st.multiselect("Choix des variables",var_cat)
                        ftp = "var/" + ','.join(var_choisi) + "/" + str(strat_remp)
                
                    if (strat_remp == "constant"):
                        cst = st.text_input("Entrez la valeur de remplacement")
                        imp = SimpleImputer(strategy = strat_remp, fill_value = cst)
                   
                    else:
                        
                        imp = SimpleImputer(strategy = strat_remp)
                   
                    if st.button("Voir et enregistrer l'opération"):
                        st.session_state.X_train[var_choisi] = imp.fit_transform(st.session_state.X_train[var_choisi])
                        st.session_state.X_test[var_choisi] = imp.transform(st.session_state.X_test[var_choisi])
                        affichage_apres_processing(str(imp) + "/var/" + ','.join(var_choisi))
               
                elif (mvo == "Supprimer les doublons"):
                
                    if st.button("Voir et enregistrer l'opération"):
                        
                        st.session_state.X_train.drop_duplicates(inplace = True)
                        st.session_state.X_test.drop_duplicates(inplace = True)
                        affichage_apres_processing("no/no")
             
                elif (mvo == "Supprimer des colonnes"):
                    
                    var_choisi = st.multiselect("Choix des variables",all_columns_names)
                    
                    if st.button("Voir et enregistrer l'opération"):
                        
                        st.session_state.X_train.drop(var_choisi, inplace=True, axis=1)
                        st.session_state.X_test.drop(var_choisi, inplace=True, axis=1)
                        affichage_apres_processing("no/no")
             
            elif (b == "Feature Selection"):
                
                st.write("En cours de création")
                cfs = st.radio("Méthode de selection des variables", ["SelectKBest","SelectPercentile","Wrapper method"])
                
                if (cfs == "SelectKBest"):
                    
                    ms = st.selectbox("Sélectionner une méthode statistique",[f_regression, f_classif, chi2, mutual_info_classif, mutual_info_regression])
                    nbf = st.number_input("Entrer le nombre de variables à garder",1,st.session_state.X_train.shape[1])
                        
                    if (st.button("Voir et enregistrer l'opération")):
                            
                        st.session_state.y_train = st.session_state.X_train[var_cible]
                        st.session_state.y_test = st.session_state.X_test[var_cible]
                        
                        st.session_state.X_train.drop(var_cible, inplace=True, axis=1)
                        st.session_state.X_test.drop(var_cible, inplace=True, axis=1)

                        selecteur = SelectKBest(ms, k=nbf).fit(st.session_state.X_train, st.session_state.y_train)
                        X_train_mat = selecteur.transform(st.session_state.X_train)
                        X_test_mat = selecteur.transform(st.session_state.X_test)

                        new_col = st.session_state.X_train.columns[selecteur.get_support()].tolist()
                        st.session_state.X_train = pd.DataFrame(X_train_mat, columns = new_col)
                        st.session_state.X_test = pd.DataFrame(X_test_mat, columns = new_col)

                        st.session_state.y_train.reset_index(drop = True, inplace = True)
                        st.session_state.y_test.reset_index(drop = True, inplace = True)

                        st.session_state.X_train[var_cible] = st.session_state.y_train
                        st.session_state.X_test[var_cible] = st.session_state.y_test

                        affichage_apres_processing("no/no")
                            
                elif (cfs == "SelectPercentile"):
                    
                    ms = st.selectbox("Sélectionner une méthode statistique",["f_regression", "f_classif", "chi2", "mutual_info_classif", "mutual_info_regression"])
                    nbf = st.number_input("Pourcentage de varaibles à garder",1,100)
                    
                    if (st.button("Voir et enregistrer l'opération")):
                        
                        st.session_state.y_train = st.session_state.X_train[var_cible]
                        st.session_state.y_test = st.session_state.X_test[var_cible]
                        
                        st.session_state.X_train.drop(var_cible, inplace=True, axis=1)
                        st.session_state.X_test.drop(var_cible, inplace=True, axis=1)

                        selecteur = SelectPercentile(ms, percentile=nbf).fit(st.session_state.X_train, st.session_state.y_train)
                        X_train_mat = selecteur.transform(st.session_state.X_train)
                        X_test_mat = selecteur.transform(st.session_state.X_test)

                        new_col = st.session_state.X_train.columns[selecteur.get_support()].tolist()
                        st.session_state.X_train = pd.DataFrame(X_train_mat, columns = new_col)
                        st.session_state.X_test = pd.DataFrame(X_test_mat, columns = new_col)

                        st.session_state.y_train.reset_index(drop = True, inplace = True)
                        st.session_state.y_test.reset_index(drop = True, inplace = True)

                        st.session_state.X_train[var_cible] = st.session_state.y_train
                        st.session_state.X_test[var_cible] = st.session_state.y_test

                        affichage_apres_processing("no/no")    
                
                elif (cfs == "Wrapper method"):
                    
                    fb = st.selectbox("Selectionne la méthode",["Backward", "Forward"])
                    ta = st.radio("Type d'algorithme", ["Classification","Regression"])
                    
                    choose_model = LogisticRegression()
                    
                    if (ta == "Classification"):
                        
                        choix_algo = st.selectbox("Choisir l'algo pour la selection de variables", ["a","b","c"])

                        if (st.button("Voir et enregistrer l'opération")):
                            
                            if (fb == "Forward"):
                                bol = True
                            else:
                                bol = False
                            
                            list_var = list(wrapper_method(LogisticRegression(), bol, 'accuracy', st.session_state.X_train.iloc[:,:-1], st.session_state.X_train[var_cible]))
                            
                            st.session_state.X_train = st.session_state.X_train[list_var + var_cible]
                            st.session_state.X_test = st.session_state.X_test[list_var + var_cible]
                            
                            affichage_apres_processing("WM/"+"var/" + ','.join(list_var) + "/" + str(fb))
                            
                    elif (ta == "Regression"):
                        
                        choix_algo = st.selectbox("Choisir l'algo pour la selection de variables", ["a","b","c"])

                        if (st.button("Voir et enregistrer l'opération")):
                            
                            if (fb == "Forward"):
                                bol = True
                            else:
                                bol = False
                            
                            list_var = list(wrapper_method(LinearRegression(), bol, 'neg_root_mean_squared_error', st.session_state.X_train.iloc[:,:-1], st.session_state.X_train[var_cible]))
                            
                            st.session_state.X_train = st.session_state.X_train[list_var + var_cible]
                            st.session_state.X_test = st.session_state.X_test[list_var + var_cible]
                            
                            affichage_apres_processing("no/no")
                    
            elif (b == "Preprocessing"):
                
                bp = st.radio("Preprocessing", ["Méthodes de normalisation","Encodage des variables catégorielles",
                                                "Discrétisation", "Feature Engineering"])
                
                if (bp == "Méthodes de normalisation"):
                    
                    mn = st.radio("Choisir une méthode de normalisation", ["StandardScaler", "MinMaxScaler", "MaxAbsScaler", "RobustScaler", 
                                                                           "QuantileTransformer", "PowerTransformer", "Normalizer"])
                    var_choisi = st.multiselect("Choix des variables",var_cont)
                    
    
                    if (mn == "StandardScaler"):
                            
                        if st.button("Voir et enregistrer l'opération"):
                                
                            scaler = StandardScaler()
                            apply_operation_prepro(scaler, var_choisi) 
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
    
                    elif(mn == "MinMaxScaler"):
                            
                        if st.button("Voir et enregistrer l'opération"):
                                
                            scaler = MinMaxScaler()
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                            
                    elif(mn == "MaxAbsScaler"):
                            
                        if st.button("Voir et enregistrer l'opération"):  
                                
                            scaler = MaxAbsScaler()
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                            
                    elif(mn == "RobustScaler"):
                            
                        if st.button("Voir et enregistrer l'opération"): 
                                
                            scaler = RobustScaler()
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                            
                    elif(mn == "QuantileTransformer"):
                        
                        nqu = st.slider("Sélectionner le nombre de quantiles (mettre 1000 par défaut)", 2, 10000)
                        od = st.radio("Choisir la distribution", ["uniform", "normal"])
                            
                        if st.button("Voir et enregistrer l'opération"):
                                
                            scaler = QuantileTransformer(n_quantiles = nqu, output_distribution = od)
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                        
                    elif(mn == "PowerTransformer"):
                        
                        mtd = st.radio("Choisir la méthode", ["yeo-johnson", "box-cox"])
                        
                        if (mtd == "box-cox"):
                            st.write("Les valeurs doivent être positives")
                        
                        if st.button("Voir et enregistrer l'opération"):     
                                
                            scaler = PowerTransformer(method = mtd)
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                        
                    elif(mn == "Normalizer"):
                            
                        nrm = st.radio("Choisir la norme", ["l1", "l2", "max"])
                        
                        if st.button("Voir et enregistrer l'opération"): 
                            
                            scaler = Normalizer(norm = nrm)
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
                
                elif (bp == "Encodage des variables catégorielles"):
                   
                    ec = st.radio("Choisir une méthode d'encodage", ["OrdinalEncoder","OneHotEncoder"])
                    var_choisi = st.multiselect("Choix des variables",var_cat)
                    
                    if (ec == "OrdinalEncoder"):
    
                        if st.button("Voir et enregistrer l'opération"):
                            
                            scaler = OrdinalEncoder()
                            apply_operation_prepro(scaler, var_choisi)
                            affichage_apres_processing(str(scaler) +"/var/" + ','.join(var_choisi))
    
                    elif(ec == "OneHotEncoder"):
                        
                        if st.button("Voir et enregistrer l'opération"):
                            
                            for j in range(len(var_choisi)):
                                
                                scaler = OneHotEncoder(sparse = False)
                                a = scaler.fit_transform(st.session_state.X_train[[var_choisi[j]]])
                                nv_var1 = []
                                b = scaler.transform(st.session_state.X_test[[var_choisi[j]]])
                                
                                for i in range(len(st.session_state.X_train[var_choisi[j]].value_counts().index)):
                                    
                                    nv_var1.append(var_choisi[j] + "-" + str(st.session_state.X_train[var_choisi[j]].value_counts().index[i]))
                                    st.session_state.X_train[nv_var1[i]] = a[:,i]
                                    st.session_state.X_test[nv_var1[i]] = b[:,i]
                                
                            st.session_state.X_train.drop(var_choisi, axis = 1, inplace = True)
                            st.session_state.X_test.drop(var_choisi, axis = 1, inplace = True)
                            affichage_apres_processing("OneHotEncoder(sparse = False)" +"/var/" + ','.join(var_choisi))
               
                elif (bp == "Discrétisation"):
                    
                    ds = st.radio("Choisir une méthode de discrétisation", ["KBinsDiscretizer","Binarizer"])
                    
                    if (ds == "KBinsDiscretizer"):
    
                            var_a_discretiser = st.multiselect("Choix des variables",var_cont)
                            bins = st.slider("Sélectionner le nombre de bins", 2, 15)
                            strategie = st.selectbox("Sélectionner la stratégie",["uniform", "quantile","kmeans"])
                            
                            if st.button("Voir et enregistrer l'opération"):
                                
                                kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy = strategie)
                                apply_operation_prepro(kbd, var_a_discretiser)
                                affichage_apres_processing(str(kbd) +"/var/" + ','.join(var_a_discretiser))
    
                    elif(ds == "Binarizer"):
                            
                            var_a_discretiser = st.multiselect("Choix des variables",var_cont)
                            thr = st.text_input("Seuil de binarisation", "entrer un nombre")
                            
                            if st.button("Voir et enregistrer l'opération"):
                                
                                bnr = Binarizer(threshold = float(thr))
                                apply_operation_prepro(bnr, var_a_discretiser)
                                affichage_apres_processing(str(bnr) + "/var/" + ','.join(var_a_discretiser))
                
                elif (bp == "Feature Engineering"):
                    
                    ds = st.radio("Choisir une méthode de feature engineering", ["Polynomial Features","Custom Transformers"])
                
                    if (ds == "Polynomial Features"):
                        
                        var_choisi = st.multiselect("Choix des variables",var_cont)
                        deg = st.slider("Sélectionner le degré", 2, 4)
                        intercation = st.radio("Choisir le mode", ["full", "interaction uniquement"])
                        
                        if st.button("Voir et enregistrer l'opération"):
                            
                            for i in range(len(obtain_list(list_polynom(var_choisi, deg)))):                               
                                st.session_state.X_train[obtain_list(list_polynom(var_choisi, deg))[i]] = calcul_pol(obtain_list(list_polynom(var_choisi, deg)),st.session_state.X_train)[i]
                                st.session_state.X_test[obtain_list(list_polynom(var_choisi, deg))[i]] = calcul_pol(obtain_list(list_polynom(var_choisi, deg)),st.session_state.X_test)[i]
                            
                            affichage_apres_processing("no/no")
                        
                    elif(ds == "Custom Transformers"):
                            
                            op = st.selectbox("Choisir un type d'opération",["Opération à une variable", "Opération entre deux variables"])
                            
                            if  (op == "Opération à une variable"):
                                
                                var_choisi = st.selectbox("Choix d'une variable",var_cont)
                                mop = st.radio("Choisir l'opération'", ["exp", "log"])
                                
                                if st.button("Voir et enregistrer l'opération"):
                                    
                                    if (mop == "exp"):
                                        
                                        transformer = FunctionTransformer(np.exp)
                                        st.session_state.X_train[mop + "_" + var_choisi] = transformer.transform(st.session_state.X_train[var_choisi])
                                        st.session_state.X_test[mop + "_" + var_choisi] = transformer.transform(st.session_state.X_test[var_choisi])
                                        affichage_apres_processing("no/no")
                                        
                                    elif (mop == "log"):
                                        
                                        transformer = FunctionTransformer(np.log)
                                        st.session_state.X_train[mop + "_" + var_choisi] = transformer.transform(st.session_state.X_train[var_choisi])
                                        st.session_state.X_test[mop + "_" + var_choisi] = transformer.transform(st.session_state.X_test[var_choisi])
                                        affichage_apres_processing("no/no")
                                
                            elif (op == "Opération entre deux variables"):
                                
                                var_choisi1 = st.selectbox("Choix de la variable 1",var_cont)
                                var_choisi2 = st.selectbox("Choix de la variable 2",var_cont)            
                                oop = st.radio("Choisir l'opération'", ["+", "-", "*", "/"])
                                nv = st.text_input("Nom de la nouvelle variable", " ")
                                
                                if st.button("Voir et enregistrer l'opération"):
                                    
                                    if (oop == "+"):
                                        st.session_state.X_train[nv] = st.session_state.X_train[var_choisi1] + st.session_state.X_train[var_choisi2]
                                        st.session_state.X_test[nv] = st.session_state.X_test[var_choisi1] + st.session_state.X_test[var_choisi2]
                                    elif (oop == "*"):
                                        st.session_state.X_train[nv] = st.session_state.X_train[var_choisi1] * st.session_state.X_train[var_choisi2]
                                        st.session_state.X_test[nv] = st.session_state.X_test[var_choisi1] * st.session_state.X_test[var_choisi2]
                                    elif (oop == "-"):
                                        st.session_state.X_train[nv] = st.session_state.X_train[var_choisi1] - st.session_state.X_train[var_choisi2]
                                        st.session_state.X_test[nv] = st.session_state.X_test[var_choisi1] - st.session_state.X_test[var_choisi2]
                                    elif (oop == "/"):
                                        st.session_state.X_train[nv] = st.session_state.X_train[var_choisi1] / st.session_state.X_train[var_choisi2]
                                        st.session_state.X_test[nv] = st.session_state.X_test[var_choisi1] / st.session_state.X_test[var_choisi2]
    
                                    affichage_apres_processing("no/no")
                                
            elif (b == "Modèle"):
                            
                mm = st.radio("Choisir le type d'algorithme", ["classification", "régression"])
                gr = st.radio("Choisir la méthode l'optimisation des hyperparamètres", ["Grille de recherche", "Manuel"])
                
                if (mm == "classification"):
                    cm = st.selectbox("Choisir un algorithme",["SVM","KNN","Random Forest", "XGBoost"])
                    
                    if (cm == "SVM"):
                        
                        if (gr == "Manuel"):
                            param_reg = st.text_input("Valeur de C, paramètre de régularisation", "Entrer valeur numérique")
                            opt = st.selectbox("Choisir un algorithme",["linear", "poly", "rbf", "sigmoid", "precomputed"])
                            
                            if (opt == "poly"):
                                
                                deg = st.slider("Degré du polynome", 2, 10)
                                
                                if st.button("Lancer l'entraînement"):
                                    
                                    op_avant_algo(var_cible)
                                    
                                    classifier = SVC(kernel = opt, C = float(param_reg), degree = int(deg))
                                    classifier.fit(st.session_state.X_train, st.session_state.y_train)
                                    y_pred = classifier.predict(st.session_state.X_test)
                                    
                                    affichage_result_classification(st.session_state.y_test, y_pred)
                                    st.session_state.listeop.append(str(classifier) + "/model")
                                    affichage_pipeline(st.session_state.listeop)
                                                                
                            else:
                                
                                if st.button("Lancer l'entraînement"):
                                    
                                    op_avant_algo(var_cible)
                                    
                                    classifier = SVC(kernel = opt, C = float(param_reg))
                                    classifier.fit(st.session_state.X_train, st.session_state.y_train)
                                    y_pred = classifier.predict(st.session_state.X_test)
                                    
                                    affichage_result_classification(st.session_state.y_test, y_pred)
                                    st.session_state.listeop.append(str(classifier) + "/model")
                                    affichage_pipeline(st.session_state.listeop)
 
                        elif (gr == "Grille de recherche"):
                            
                            cv = st.slider("Sélectionner le nombre de CVs", 2, 8)
                            krn = st.multiselect("Choisir les noyaux à tester", ["linear", "rbf", "sigmoid", "precomputed"])
                            val_reg = st.text_input("Ajouter une valeur de regularisation", "Ex : 0.01")
                            
                            if (st.button("Ajouter la valeur")):
                                
                                st.session_state.l1_gp.append(float(val_reg))
                                st.write(st.session_state.l1_gp)
                                                    
                            if st.button("Lancer l'entraînement"):
                                
                                parameters = {'kernel': krn, 'C' : st.session_state.l1_gp}
                                    
                                op_avant_algo(var_cible)                               

                                svc = SVC()
                                clf = GridSearchCV(svc, parameters, cv = cv)
                                clf.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = clf.best_params_
                                st.text(best_param)
                                
                                y_pred = clf.predict(st.session_state.X_test) 
                                affichage_result_classification(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(SVC(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "KNN"):
                        
                        if (gr == "Manuel"):
                            
                            n_neighbors = st.number_input("Nombre de voisins", 2, 20)
                            algo = st.selectbox("Choisir un algorithme",["auto", "ball_tree", "kd_tree", "brute"])
                            weights = st.selectbox("Weight fonction", ["uniform", "distance"])
                                
                            if st.button("Lancer l'entraînement"):
                                    
                                op_avant_algo(var_cible)
                                    
                                classifier = KNeighborsClassifier(n_neighbors = n_neighbors, algorithm = algo, weights = weights)
                                classifier.fit(st.session_state.X_train, st.session_state.y_train)
                                y_pred = classifier.predict(st.session_state.X_test)
                                
                                affichage_result_classification(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(classifier) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                            
                        elif (gr == "Grille de recherche"):
                            
                            algo = st.multiselect("Choisir un algorithme",["auto", "ball_tree", "kd_tree", "brute"])
                            n_neighbors = st.number_input("Nombre de voisins", 2, 20)
                                
                            if (st.button("Ajouter la valeur")):
                                    
                                st.session_state.l1_gp.append(n_neighbors)
                                st.write(st.session_state.l1_gp)
                                                        
                            if st.button("Lancer l'entraînement"):
                                    
                                parameters = {'algorithm': algo, 'n_neighbors' : st.session_state.l1_gp}
                                cv = 3
                                        
                                op_avant_algo(var_cible)                               
    
                                knn = KNeighborsClassifier()
                                clf = GridSearchCV(knn, parameters, cv = cv)
                                clf.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = clf.best_params_
                                st.text(best_param)
                                
                                y_pred = clf.predict(st.session_state.X_test) 
                                affichage_result_classification(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(KNeighborsClassifier(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "Random Forest"):
                        
                        if (gr == "Manuel"):
                            
                            n_estimators = st.number_input("Nombre d'arbres de décision", 10, max_value=1000)
                            max_depth = st.number_input("Profondeur d'un arbre", 2, 8)

                            if st.button("Lancer l'entraînement"):
                                    
                                op_avant_algo(var_cible)
                                    
                                classifier = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth)
                                classifier.fit(st.session_state.X_train, st.session_state.y_train)
                                y_pred = classifier.predict(st.session_state.X_test)
                                
                                affichage_result_classification(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(classifier) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                            
                        elif (gr == "Grille de recherche"):
                            
                            n_estimators = st.number_input("Nombre d'arbres de décision", 10, 1000)
                                
                            if (st.button("Ajouter la valeur n_estimators")):
                                    
                                st.session_state.l1_gp.append(n_estimators)
                                st.write(st.session_state.l1_gp)
                                
                            max_depth = st.number_input("Profondeur d'un arbre", 2, 10)
                            
                            if (st.button("Ajouter la valeur max_depth")):
                                    
                                st.session_state.l2_gp.append(max_depth)
                                st.write(st.session_state.l2_gp)
                                                        
                            if st.button("Lancer l'entraînement"):
                                    
                                parameters = {'n_estimators': st.session_state.l1_gp, 'max_depth' : st.session_state.l2_gp}
                                        
                                op_avant_algo(var_cible)
                                
                                rff = RandomForestClassifier()
                                clf = GridSearchCV(rff, parameters, cv = cv)
                                clf.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = clf.best_params_
                                st.text(best_param)
                                
                                y_pred = clf.predict(st.session_state.X_test) 
                                affichage_result_classification(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(RandomForestClassifier(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "XGBoost"):
                        
                        if st.button("Lancer l'entraînement avec optimisation baysienne"):
                                
                            op_avant_algo(var_cible)  
                                
                            bayes_cv_tuner = BayesSearchCV(
                                estimator = XGBClassifier(
                                    n_jobs = 1,
                                    objective = 'binary:logistic',
                                    eval_metric = 'auc',
                                    tree_method='approx'
                                ),
                                search_spaces = {
                                    'learning_rate': (0.01, 1.0, 'log-uniform'),
                                    'max_depth': (2, 10),
                                    'n_estimators': (30, 1000),
                                    'max_delta_step': (0, 20),
                                    'subsample': (0.01, 1.0, 'uniform'),
                                    'colsample_bytree': (0.01, 1.0, 'uniform'),
                                    'colsample_bylevel': (0.01, 1.0, 'uniform'),
                                    'min_child_weight': (0, 10),
                                    'scale_pos_weight': (1, 500, 'log-uniform'),
                                    'reg_lambda': (0.5, 100, 'log-uniform'),
                                    'reg_alpha': (0.5, 100, 'log-uniform'),
                                    'gamma' : (0.5, 100, 'log-uniform')
                                },    
                                scoring = 'roc_auc',
                                cv = StratifiedKFold(
                                    n_splits=2,
                                    shuffle=True,
                                    random_state=42
                                ),
                                n_jobs = 4,
                                n_iter = 5,   
                                verbose = 0,
                                refit = True,
                                random_state = 42
                            )
                            
                            
                            result = bayes_cv_tuner.fit(st.session_state.X_train, st.session_state.y_train)
                            best_params = pd.Series(result.best_params_)
                            st.write("Meilleur paramètre :")
                            best_param = dict(list(bayes_cv_tuner.best_params_.items()))
                            st.text(best_param)

                            st.write("ROC-AUC :")
                            st.write(np.round(result.best_score_, 4))
                            
                            y_pred = result.predict(st.session_state.X_test) 
                            affichage_result_classification(st.session_state.y_test, y_pred)
                            st.session_state.listeop.append("XGBClassifier" + "/model/" + str(best_param))
                            affichage_pipeline(st.session_state.listeop)
                            
                                    
                elif (mm == "régression"):
                    cm = st.selectbox("Choisir un algorithme",["SVR","KNN","Random Forest", "XGBoost"])
                    
                    if (cm == "SVR"):
                        
                        if (gr == "Manuel"):
                            param_reg = st.text_input("Valeur de C, paramètre de régularisation", "Entrer valeur numérique")
                            opt = st.selectbox("Choisir un algorithme",["linear", "poly", "rbf", "sigmoid", "precomputed"])
                            
                            if (opt == "poly"):
                                
                                deg = st.slider("Degré du polynome", 2, 10)
                                
                                if st.button("Lancer l'entraînement"):
                                    
                                    op_avant_algo(var_cible)
                                    
                                    regressor = SVR(kernel = opt, C = float(param_reg), degree = int(deg))
                                    regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                    y_pred = regressor.predict(st.session_state.X_test)
                                    
                                    affichage_result_regression(st.session_state.y_test, y_pred)
                                    st.session_state.listeop.append(str(regressor) + "/model")
                                    affichage_pipeline(st.session_state.listeop)
                                                                
                            else:
                                
                                if st.button("Lancer l'entraînement"):
                                    
                                    op_avant_algo(var_cible)
                                    
                                    regressor = SVR(kernel = opt, C = float(param_reg))
                                    regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                    y_pred = regressor.predict(st.session_state.X_test)
                                    
                                    affichage_result_regression(st.session_state.y_test, y_pred)
                                    st.session_state.listeop.append(str(regressor) + "/model")
                                    affichage_pipeline(st.session_state.listeop)
 
                        elif (gr == "Grille de recherche"):
                            
                            cv = st.slider("Sélectionner le nombre de CVs", 1, 8)
                            krn = st.multiselect("Choisir les noyaux à tester", ["linear", "rbf", "sigmoid", "precomputed"])
                            val_reg = st.text_input("Ajouter une valeur de regularisation", "Ex : 0.01")
                            
                            if (st.button("Ajouter la valeur")):
                                
                                st.session_state.l1_gp.append(float(val_reg))
                                st.write(st.session_state.l1_gp)
                                                    
                            if st.button("Lancer l'entraînement"):
                                
                                parameters = {'kernel': krn, 'C' : st.session_state.l1_gp}
                                    
                                op_avant_algo(var_cible)                               

                                svc = SVR()
                                regressor = GridSearchCV(svc, parameters, cv = cv)
                                regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = regressor.best_params_
                                st.text(best_param)
                                
                                y_pred = regressor.predict(st.session_state.X_test) 
                                affichage_result_regression(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(SVR(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "KNN"):
                        
                        if (gr == "Manuel"):
                            
                            n_neighbors = st.number_input("Nombre de voisins", 2, 20)
                            algo = st.selectbox("Choisir un algorithme",["auto", "ball_tree", "kd_tree", "brute"])
                            weights = st.selectbox("Weight fonction", ["uniform", "distance"])
                                
                            if st.button("Lancer l'entraînement"):
                                    
                                op_avant_algo(var_cible)
                                    
                                regressor = KNeighborsRegressor(n_neighbors = n_neighbors, algorithm = algo, weights = weights)
                                regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                y_pred = regressor.predict(st.session_state.X_test)
                                
                                affichage_result_regression(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(regressor) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                            
                        elif (gr == "Grille de recherche"):
                            
                            algo = st.multiselect("Choisir un algorithme",["auto", "ball_tree", "kd_tree", "brute"])
                            n_neighbors = st.number_input("Nombre de voisins", 2, 20)
                                
                            if (st.button("Ajouter la valeur")):
                                    
                                st.session_state.l1_gp.append(n_neighbors)
                                st.write(st.session_state.l1_gp)
                                                        
                            if st.button("Lancer l'entraînement"):
                                    
                                parameters = {'algorithm': algo, 'n_neighbors' : st.session_state.l1_gp}
                                cv = 3
                                        
                                op_avant_algo(var_cible)                               
    
                                knnr = KNeighborsRegressor()
                                regressor = GridSearchCV(knnr, parameters, cv = cv)
                                regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = regressor.best_params_
                                st.text(best_param)
                                
                                y_pred = regressor.predict(st.session_state.X_test) 
                                affichage_result_regression(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(KNeighborsRegressor(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "Random Forest"):
                        
                        if (gr == "Manuel"):
                            
                            n_estimators = st.number_input("Nombre d'arbres de décision", 10, 1000)
                            max_depth = st.number_input("Profondeur d'un arbre", 2, 8)

                            if st.button("Lancer l'entraînement"):
                                    
                                op_avant_algo(var_cible)
                                    
                                regressor = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth)
                                regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                y_pred = regressor.predict(st.session_state.X_test)
                                
                                affichage_result_regression(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(regressor) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                            
                        elif (gr == "Grille de recherche"):
                            
                            n_estimators = st.number_input("Nombre d'arbres de décision", 10, 1000)
                                
                            if (st.button("Ajouter la valeur n_estimators")):
                                    
                                st.session_state.l1_gp.append(n_estimators)
                                st.write(st.session_state.l1_gp)
                                
                            max_depth = st.number_input("Profondeur d'un arbre",2,10)
                            
                            if (st.button("Ajouter la valeur max_depth")):
                                    
                                st.session_state.l2_gp.append(max_depth)
                                st.write(st.session_state.l2_gp)
                                                        
                            if st.button("Lancer l'entraînement"):
                                    
                                parameters = {'n_estimators': st.session_state.l1_gp, 'max_depth' : st.session_state.l2_gp}
                                        
                                op_avant_algo(var_cible)                                 
    
                                rfr = RandomForestRegressor()
                                regressor = GridSearchCV(rfr, parameters, cv = cv)
                                regressor.fit(st.session_state.X_train, st.session_state.y_train)
                                best_param = regressor.best_params_
                                st.text(best_param)
                                
                                y_pred = regressor.predict(st.session_state.X_test) 
                                affichage_result_regression(st.session_state.y_test, y_pred)
                                st.session_state.listeop.append(str(RandomForestRegressor(**best_param)) + "/model")
                                affichage_pipeline(st.session_state.listeop)
                                
                    elif (cm == "XGBoost"):
                        
                        if st.button("Lancer l'entraînement avec optimisation baysienne"):
                                
                            op_avant_algo(var_cible)   
                                
                            bayes_cv_tuner = BayesSearchCV(
                                estimator = XGBRegressor(),
                                search_spaces = {
                                    'learning_rate': (0.01, 1.0, 'log-uniform'),
                                    'max_depth': (2, 10),
                                    'n_estimators': (30, 1000),
                                    'max_delta_step': (0, 20),
                                    'subsample': (0.01, 1.0, 'uniform'),
                                    'colsample_bytree': (0.01, 1.0, 'uniform'),
                                    'colsample_bylevel': (0.01, 1.0, 'uniform'),
                                    'min_child_weight': (0, 10),
                                    'scale_pos_weight': (1, 500, 'log-uniform'),
                                    'reg_lambda': (0.5, 100, 'log-uniform'),
                                    'reg_alpha': (0.5, 100, 'log-uniform'),
                                    'gamma' : (0.5, 100, 'log-uniform')
                                },    
                                scoring = 'neg_mean_squared_error',
                                cv = StratifiedKFold(
                                    n_splits=3,
                                    shuffle=True,
                                    random_state=42
                                ),
                                n_jobs = 4,
                                n_iter = 5,   
                                verbose = 0,
                                refit = True,
                                random_state = 42
                            )
                            
                            result = bayes_cv_tuner.fit(st.session_state.X_train, st.session_state.y_train)
                            best_params = pd.Series(result.best_params_)
                            st.write("Meilleur paramètre :")
                            best_param = dict(list(bayes_cv_tuner.best_params_.items()))
                            st.text(best_param)
                            
                            y_pred = result.predict(st.session_state.X_test) 
                            affichage_result_regression(st.session_state.y_test, y_pred)
                            st.session_state.listeop.append("XGBRegressor" + "/model/" + str(best_param))
                            affichage_pipeline(st.session_state.listeop)
