# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.cluster import KMeans
from zipfile import ZipFile
import plotly.express as px
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import pickle
#import shap
import plotly.graph_objects as go
#import sklearn.metrics
plt.style.use('fivethirtyeight')

def main():
     @st.cache
     def load_data():
         z = ZipFile("default_risk.zip")
         data = pd.read_csv(z.open('default_risk.csv'),
                            index_col='SK_ID_CURR', encoding ='utf-8')

         z = ZipFile("X_data.zip")
         sample = pd.read_csv(z.open('X_data.csv'),
                              index_col='SK_ID_CURR', encoding ='utf-8')

         description = pd.read_csv("features_description.csv",
                                   usecols=['Row', 'Features'], index_col=0, encoding= 'unicode_escape')

         #target =  data.iloc[:, -1:]
         #target = pd.read_csv("TARGET.csv")
         #return data, sample, target, description
         return data, sample, description

     def load_model():
            '''loading the trained model'''
            pickle_in = open('LGBMClassifier.pkl', 'rb')
            clf = pickle.load(pickle_in)
            return clf

     @st.cache(allow_output_mutation=True)
     def load_knn(sample):
            knn = knn_training(sample)
            return knn

     @st.cache
     def load_infos_gen(data):
            lst_infos = [data.shape[0],
                         round(data["AMT_INCOME_TOTAL"].mean(), 2),
                         round(data["AMT_CREDIT"].mean(), 2)]

            nb_credits = lst_infos[0]
            rev_moy = lst_infos[1]
            credits_moy = lst_infos[2]

            #targets = data.TARGET.value_counts()
            targets = data["TARGET"].value_counts()

            return nb_credits, rev_moy, credits_moy, targets

     def identite_client(data, id):
            data_client = data[data.index == int(id)]
            return data_client

     @st.cache
     def load_age_population(data):
            data_age = round((data["DAYS_BIRTH"]/-365), 2)
            return data_age

     @st.cache
     def load_income_population(sample):
            df_income = pd.DataFrame(sample["AMT_INCOME_TOTAL"])
            df_income = df_income.loc[df_income['AMT_INCOME_TOTAL'] < 200000, :]
            return df_income

     @st.cache
     def load_prediction(sample, id, clf):
            X = sample.iloc[:, :-1]
            score = clf.predict_proba(X[X.index == int(id)])[:, 1]
            return score

     @st.cache
     def load_kmeans(sample, id, mdl):
            index = sample[sample.index == int(id)].index.values
            index = index[0]
            data_client = pd.DataFrame(sample.loc[sample.index, :])
            df_neighbors = pd.DataFrame(knn.fit_predict(data_client), index=data_client.index)
            df_neighbors = pd.concat([df_neighbors, data], axis=1)
            return df_neighbors.iloc[:, 1:].sample(10)

     @st.cache
     def knn_training(sample):
            knn = KMeans(n_clusters=2).fit(sample)
            return knn

     @st.cache
     def load_probabilities(sample, id, clf):
           index = sample[sample.index == int(id)].index.values
           index = index[0]
           X = sample.iloc[:, :-1]
           score = clf.predict_proba(X[X.index == int(id)])[:, 1]
           df_prob = pd.DataFrame(clf.predict_proba(X)[:, -1], index=X.index)
           #df_prob = df_prob.reset_index()
           df_prob.columns = ['Default Probability']
           
           df_prob['prob_rating'] = ['Worse' if x > score else 'Equal or Better' for x in df_prob['Default Probability']]
           #percount = df_prob['greater'].sum()
                      
           position=df_prob[X.index == int(id)].index[0]    #1st condition fullfilled 
           
           resultdf=df_prob.loc[position-5:position+5,:] 
           
           #df_prob = pd.concat([df_prob, data], axis=1)
           #return df_prob.iloc[:, 1:].sample(5)
           return resultdf, df_prob

        # Loading data……
     #data, sample, target, description = load_data()
     data, sample, description = load_data()
     id_client = sample.index.values
     clf = load_model()

        #######################################
        # SIDEBAR
        #######################################

        # Title display
     html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
     st.markdown(html_temp, unsafe_allow_html=True)

        # Customer ID selection
     st.sidebar.header("**General Info**")

        # Loading selectbox
     chk_id = st.sidebar.selectbox("Client ID", id_client)

        # Loading general info
     nb_credits, rev_moy, credits_moy, targets = load_infos_gen(data)

        ### Display of information in the sidebar ###
        # Number of loans in the sample
     st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
     st.sidebar.text(nb_credits)

        # Average income
     st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
     st.sidebar.text(rev_moy)

        # AMT CREDIT
     st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
     st.sidebar.text(credits_moy)

        # PieChart
        #st.sidebar.markdown("<u>......</u>", unsafe_allow_html=True)
     fig, ax = plt.subplots(figsize=(5, 5))
     plt.pie(targets, explode=[0, 0.1], labels=['No default', 'Default'], autopct='%1.1f%%', startangle=90)
     st.sidebar.pyplot(fig)

        #######################################
        # HOME PAGE - MAIN CONTENT
        ######################################
    
     if st.checkbox("Add file of Processed new customers:"):
         uploaded_file = st.file_uploader("Choose a file")
         if uploaded_file is not None:
             # Can be used wherever a "file-like" object is accepted:
             dataframe = pd.read_csv(uploaded_file)
             #st.write(dataframe)
             id_client2 = dataframe.index.values
             #clf = load_model()
             chk_id = st.selectbox("Client ID", id_client2)
             
             prediction4 = load_prediction(dataframe, chk_id, clf)
             st.write("**Default probability : **{:.0f} %".format(round(float(prediction4)*100, 2)))
             
             #daq.Gauge(
              #   color={"gradient":True,"ranges":{"green":[0,0.2],"yellow":[0.2,0.4],"red":[0.4,1.0]}},
              #   value= prediction,
              #   label='Default Scale',
              #   max=1.0,
              #   min=0.0,
             #)
              
             fig2 = go.Figure(go.Indicator(
                 mode = "gauge+number",
                 value = round(float(prediction4)*100, 2),
                 domain = {'x': [0, 0.6], 'y': [0, 0.6]},
                 title = {'text': "Default Probability Scale", 'font': {'size': 21}},
                 number= {"font":{"size":20}, "suffix": "%"},
                 gauge = {'axis': {'range': [None, 100]},
                          'steps' : [
                              {'range': [0, 10], 'color': "white"},
                              {'range': [10, 30], 'color': "yellow"},
                              {'range': [30, 100], 'color': "red"}]                  
                          }))
        
             st.plotly_chart(fig2)
             
             prediction5, prob_comp1 = load_probabilities(dataframe, chk_id, clf)
             
             fig, ax = plt.subplots(figsize=(10, 5))
             sns.histplot(prob_comp1['Default Probability'], edgecolor= 'k', color="goldenrod", bins=20)
             ax.axvline(float(prediction4), color="green", linestyle='--')
             ax.set(title='Default Probability Distribution', xlabel='Default Probability', ylabel='')
             st.pyplot(fig)
             
             # Feature importance / description
             if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
                 shap.initjs()
                 X = dataframe.iloc[:, :-1]
                 X = X[X.index == chk_id]
                 number = st.slider("Pick a number of features…", 0, 20, 5)
     
                 fig, ax = plt.subplots(figsize=(10, 10))
                 explainer = shap.TreeExplainer(load_model())
                 shap_values = explainer.shap_values(X)
                 shap.summary_plot(shap_values[0], X, plot_type="bar", max_display=number, color_bar=False, plot_size=(5, 5))
                 st.pyplot(fig)
     
                 if st.checkbox("Need help about feature description ?"):
                     list_features = description.index.to_list()
                     feature = st.selectbox('Feature checklist…', list_features)
                     st.table(description.loc[description.index == feature][:1])
     
             else:
                 st.markdown("<i>…</i>", unsafe_allow_html=True)
     else:
               #st.markdown("<i>…</i>", unsafe_allow_html=True)
    
        
        
            # Display Customer ID from Sidebar
          st.write("Customer ID selection :", chk_id)
    
            # Customer information display : Customer Gender, Age, Family status, Children, …
          st.header("**Customer information display**")
         
          prediction = load_prediction(sample, chk_id, clf)
        # st.write("**Default probability : **{:.0f} %".format(round(float(prediction)*100, 2)))
    
          fig2 = go.Figure(go.Indicator(
             mode = "gauge+number", 
             value = round(float(prediction)*100,2),
             domain = {'x': [0, 0.7], 'y': [0, 0.7]},
             title = { 'text': "Default probability gauge", 'font': {'size': 25}},
             number = {'font': {'size': 20}, "suffix": "%"},
             gauge = {'axis': {'range': [None, 100]}, 
                      'steps' : [
                         {'range': [0,10], 'color': "white"},
                         {'range': [10,30], 'color': "yellow"},
                         {'range': [30,100], 'color': "rgb(236,66,32)"}]
                     }))
    
          st.plotly_chart(fig2)
    
    
          if st.checkbox("Show customer information ?"):
    
                infos_client = identite_client(data, chk_id)
                st.write("**Gender : **", infos_client["CODE_GENDER"].values[0])
                st.write("**Age : **{:.0f} ans".format(int(infos_client["DAYS_BIRTH"]/-365)))
                st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS"].values[0])
                st.write("**Number of children : **{:.0f}".format(infos_client["CNT_CHILDREN"].values[0]))
    
                # Age distribution plot
                data_age = load_age_population(data)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data_age, edgecolor= 'k', color="goldenrod", bins=20)
                ax.axvline(int(infos_client["DAYS_BIRTH"].values / -365), color="green", linestyle='--')
                ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
                st.pyplot(fig)
    
                st.subheader("*Income (USD)*")
                st.write("**Income total : **{:.0f}".format(infos_client["AMT_INCOME_TOTAL"].values[0]))
                st.write("**Credit amount : **{:.0f}".format(infos_client["AMT_CREDIT"].values[0]))
                st.write("**Credit annuities : **{:.0f}".format(infos_client["AMT_ANNUITY"].values[0]))
                st.write("**Amount of property for credit : **{:.0f}".format(infos_client["AMT_GOODS_PRICE"].values[0]))
    
                # Income distribution plot
                data_income = load_income_population(data)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.histplot(data_income["AMT_INCOME_TOTAL"], edgecolor= 'k', color="goldenrod", bins=10)
                ax.axvline(int(infos_client["AMT_INCOME_TOTAL"].values[0]), color="green", linestyle='--')
                ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
                st.pyplot(fig)
    
                # Relationship Age / Income Total interactive plot
                data_sk = data.reset_index(drop=False)
                data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH']/-365).round(1)
                fig, ax = plt.subplots(figsize=(10, 10))
                fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                                 size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                                 hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
    
                fig.update_layout({'plot_bgcolor': '#f0f0f0'}, 
                                  title={'text': "Relationship Age / Income Total", 'x':0.5, 'xanchor': 'center'}, 
                                  title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))
    
                fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
                fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                                 title="Age", title_font=dict(size=18, family='Verdana'))
                fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                                 title="Income Total", title_font=dict(size=18, family='Verdana'))
    
                st.plotly_chart(fig)
    
          else:
                 st.markdown("<i>…</i>", unsafe_allow_html=True)
    
            # Customer solvability display
          st.header("**Customer file analysis**")
                # prediction = load_prediction(sample, chk_id, clf)
          st.write("**Default probability : **{:.0f} %".format(round(float(prediction)*100, 2)))
            
                 #fig2 = go.Figure(go.Indicator(
                   #  mode = "gauge+number", 
                    # value = round(float(prediction)*100,2),
                     #domain = {'x': [0, 0.7], 'y': [0, 0.7]},
                    # title = { 'text': "Default probability gauge", 'font': {'size': 25}},
                   #  number = {'font': {'size': 20}, "suffix": "%"},
                    # gauge = {'axis': {'range': [None, 100]}, 
                    #          'steps' : [
                  #               {'range': [0,10], 'color': "white"},
                    #             {'range': [10,30], 'color': "yellow"},
                  #               {'range': [30,100], 'color': "rgb(236,66,32)"}]
                 #            }))
            
              #   st.plotly_chart(fig2)
                    # Compute decision according to the best threshold
                    # if prediction <= xx :
                    #    decision = "<font color='green'>**LOAN GRANTED**</font>"
                    # else:
                    #    decision = "<font color='red'>**LOAN REJECTED**</font>"
            
                    #st.write("**Decision** *(with threshold xx%)* **: **", decision, unsafe_allow_html=True)
            
          st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
          st.write(identite_client(data, chk_id))
                 
                 # Customer Comparaison display
                 #st.header("**Probability chart**")     
                # st.markdown("<u>List of the 10 files closest to this Customer default profile :</u>", unsafe_allow_html=True)
          prediction2, prob_comp = load_probabilities(sample, chk_id, clf)
                 #st.dataframe(load_probabilities(sample, chk_id, clf))
                 #st.dataframe(prediction2)
                 
                 # Default Probability Distribution plot
          fig, ax = plt.subplots(figsize=(10, 5))
          sns.histplot(prob_comp['Default Probability'], edgecolor= 'k', color="goldenrod", bins=20)
          ax.axvline(float(prediction), color="green", linestyle='--')
          ax.set(title='Default Probability Distribution', xlabel='Default Probability ', ylabel='')
          st.pyplot(fig) 
       
            # Probability comparaison plot
          fig, ax = plt.subplots(figsize=(10, 5))
          sns.histplot(prob_comp['prob_rating'], edgecolor= 'k', color="goldenrod")
          #sns.histplot(prob_comp['prob_rating'], edgecolor= 'k', color="goldenrod", bins=2)
          ax.set(title='Default Probability Relative to Data', xlabel='', ylabel='Count')
          st.pyplot(fig)
                 
               
               # Feature importance / description
          if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
                   shap.initjs()
                   X = sample.iloc[:, :-1]
                   X = X[X.index == chk_id]
                   number = st.slider("Pick a number of features…", 0, 20, 5)
       
                   fig, ax = plt.subplots(figsize=(10, 10))
                   explainer = shap.TreeExplainer(load_model())
                   shap_values = explainer.shap_values(X)
                   shap.summary_plot(shap_values[0], X, plot_type="bar", max_display=number, color_bar=False, plot_size = (15, 15))           
                   st.pyplot(fig)
       
                   if st.checkbox("Need help about feature description ?"):
                       list_features = description.index.to_list()
                       feature = st.selectbox('Feature checklist…', list_features)
                       st.table(description.loc[description.index == feature][:1])
       
          else:
                   st.markdown("<i>…</i>", unsafe_allow_html=True)
       
       
               # Similar customer files display
          chk_voisins = st.checkbox("Show similar customer files ?")
       
          if chk_voisins:
                   knn = load_knn(sample)
                   st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
                   st.dataframe(load_kmeans(sample, chk_id, knn))
                   st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
          else:
                   st.markdown("<i>…</i>", unsafe_allow_html=True)
       
          st.markdown('***')
          st.markdown("Thanks for going through this Web App with me! I'd love feedback on this, so if you want to reach out you can find me on [twitter] (https://twitter.com/nalron_) or my [website](https://nalron.com/). *Code from [Github](https://github.com/nalron/project_credit_scoring_model)* ❤️")


if __name__ == '__main__':
    main()

