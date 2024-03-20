import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pickle
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import numpy as np


st. set_page_config(layout="wide")


def arima(train, dep, indep, arima_order):
    print(dep)
    print(indep)
    print(arima_order)
    print(type(train[dep]))
    print(type(train[indep]))
    model = ARIMA(endog=train[dep], exog=train[indep], order=arima_order)  # Example order, tune as needed
    model_fit = model.fit()
    # Save the model to a file
    with open('arima_model.pkl', 'wb') as f:
        pickle.dump(model_fit, f)
    return model_fit
  
 

def annModel(train, dep, indep):
    ann = Sequential()
    #adding the input and first hidden layer
    ann.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform',input_dim=len(indep)))
    #adding second layer
    ann.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
    #adding the output layer
    ann.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    ann.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    ann.fit(train[indep], train[dep],
        batch_size=35,
        epochs=50,
        shuffle=True,
    )

    ann.save("ann.h5")

    return ann    
    
def test(y_test):
    logr=pickle.load(open('logr.sav', 'rb'))
    rfc=pickle.load(open('rfc.sav', 'rb'))
    knn=pickle.load(open('knn.sav', 'rb'))
    svm=pickle.load(open('svm.sav', 'rb'))
    ann=load_model('ann.h5')

    st.write("Logistic Regression : ",logr.predict(y_test))
    st.write("Random Forest Classifier : ",rfc.predict(y_test))
    st.write("K Nearest Neighbors : ",knn.predict(y_test))
    st.write("Support Vector Machine : ",svm.predict(y_test))
    st.write("Artificial Neural Network : ",ann.predict(y_test))


def selectCatVar(df):
    l=[]
    for col in df.columns:
        l.append(col)

    indep = st.multiselect(
        'Select independent variables',
        l
    )

    failure = st.selectbox('Select failure field',
        l
    )

    user_input = st.radio("Is failure field a categorical variable. \nSelect yes or no:", ("yes", "no"))

    cat = st.multiselect(
        'Select other categorical variables',
        l
    )

    visualize_l = st.selectbox('Select variable to visualize against failure',
        l
    )

    return indep,failure,user_input,cat,visualize_l


st.markdown("# Analytics")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    indep,failure,user_input,cat,visualize_l=selectCatVar(df)

    df['datetime_new'] = pd.to_datetime(
        df['datetime'],
        format="%Y-%m-%d %H:%M:%S"
    )

    new_failure=None

    if user_input == "yes":
        new_failure = failure + '_new'
        df[failure]=df[failure].fillna('a')
        print(df[failure].unique())
        df[new_failure]=LabelEncoder().fit_transform(df[failure]) 
        print(df[new_failure].unique())

    for option in cat:
        new_option = option + '_new'
        df[option]=df[option].fillna('a')
        print(df[option].unique())
        df[new_option]=LabelEncoder().fit_transform(df[option]) 
        print(df[new_option].unique())
        

    #with col1:
    summary=st.button('Summary')
    visualize=st.button('Visualize your data')
    models=st.button('Models')
    trends=st.button('Seasonal/trends')
    failures_overtime=st.button('Cumulative failures')
    
    size= int(len(df)*0.75)
    train, test = df[0:size], df[size:len(df)]

    if summary:
        st.write(df.describe())
    
    if visualize:
        print("test-1")
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        dates = matplotlib.dates.date2num(df['datetime_new'])
        
        plt.plot_date(dates, df[visualize_l], linestyle='-', marker='', label=visualize_l)

        # Plot Failures
        plt.plot_date(dates, df['failure'], color='red', linestyle='None', marker='*', markersize=1, label='Failures')

        # Title and labels
        plt.title(f"Failures with {visualize_l} Variations")
        plt.xlabel("Date")
        plt.ylabel(visualize_l)


        st.pyplot(fig)
    
    if trends:

        df_copy=df[['datetime_new', 'failure_new']]
        print(type(df_copy['datetime_new']))
        ts_failures_daily = df_copy.resample('D', on='datetime_new').sum()
        result = seasonal_decompose(ts_failures_daily, model='additive')
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))

        result.observed.plot(ax=ax1)
        ax1.set_ylabel('Observed')

        result.trend.plot(ax=ax2)
        ax2.set_ylabel('Trend')

        result.seasonal.plot(ax=ax3)
        ax3.set_ylabel('Seasonal')

        result.resid.plot(ax=ax4)
        ax4.set_ylabel('Residual')

        plt.tight_layout()
        st.pyplot(fig)
    
    if failures_overtime:

        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(10)

        df_copy=df[['datetime_new', 'failure_new']]

        df_copy['fail_count'] = 0
        df_copy.loc[df_copy['failure_new'] != 0, 'fail_count'] = 1

        print(df_copy['fail_count'].unique())
        df_copy.drop(columns='failure_new')
        ts_failures_daily = df_copy.resample('D', on='datetime_new').sum()




        hist_daily_failures = ts_failures_daily['fail_count'].cumsum()

        plt.plot(hist_daily_failures.index, hist_daily_failures.values, marker='o', linestyle='-')
        plt.title('Cumulative Sum Values Over Months')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Sum')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        st.pyplot(fig)

    if models:
        fig, ax = plt.subplots()
        fig.set_figheight(1)
        fig.set_figwidth(2)
        col1, col2= st.columns([3,3])
        
        with col1:
            st.write("ARIMA")
            
            model_fit = arima(train, new_failure, indep, (1,0,1))
            b_pred = model_fit.forecast(steps=len(test[new_failure]), exog=test[indep])
            new_b_pred=[]

            for i in b_pred:
                if i >= 0.5:
                    i = 1
                    new_b_pred.append(i)
                else:
                    i = 0
                    new_b_pred.append(i)

            print(list(set(new_b_pred)))
            print("------------")
            thresholded_output = [1 if x > 0 else 0 for x in test[new_failure]]
            print(list(set(thresholded_output)))

            acc = accuracy_score(thresholded_output,new_b_pred)
            rec = recall_score(thresholded_output,new_b_pred)
            mse = mean_squared_error(test[new_failure], b_pred)

            st.write("Accuracy : ", acc)
            st.write("Recall : ", rec)
            st.write("MSE : ", mse)
        
        with col2:
            st.write("ANN")

            ann=annModel(train, new_failure, indep)

            b_pred=ann.predict(test[indep])
            b_pred_copy=b_pred
            for i in range(len(b_pred)):
                if b_pred[i] >= 0.5:
                    b_pred[i] = 1
                else:
                    b_pred[i] = 0


            thresholded_output = [1 if x > 0 else 0 for x in test[new_failure]]
            print(list(set(thresholded_output)))

            acc = accuracy_score(thresholded_output,new_b_pred)
            rec = recall_score(thresholded_output,new_b_pred)
            mse = mean_squared_error(test[new_failure], b_pred_copy)

            st.write("Accuracy : ", acc)
            st.write("Recall : ", rec)
            st.write("MSE : ", mse)
