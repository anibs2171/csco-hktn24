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


st. set_page_config(layout="wide")


def rfc(df, dep, keys):
    from sklearn.model_selection import train_test_split as tts  
    A_train, A_test, b_train, b_test = tts(df[keys], df[dep], test_size = 0.3, random_state = 0)  
    from sklearn.ensemble import RandomForestClassifier as RFC  
    rfc = RFC()  
    rfc.fit(A_train, b_train)

    b_pred = rfc.predict(A_test)
    score = accuracy_score(b_test, b_pred)
    filename = 'rfc.sav'
    pickle.dump(rfc, open(filename, 'wb'))
    return accuracy_score(b_test,b_pred), recall_score(b_test,b_pred)

    
def logregr(df, dep, keys):
    from sklearn.model_selection import train_test_split as tts  
    A_train, A_test, b_train, b_test = tts(df[keys], df[dep], test_size = 0.3, random_state = 0)  
    from sklearn.linear_model import LogisticRegression
    logr = LogisticRegression()
    logr.fit(A_train, b_train)
    
    b_pred = logr.predict(A_test)  
    filename = 'logr.sav'
    pickle.dump(logregr, open(filename, 'wb'))
    return accuracy_score(b_test,b_pred), recall_score(b_test,b_pred)
 

def knn(df, dep, keys):
    from sklearn.model_selection import train_test_split as tts  
    A_train, A_test, b_train, b_test = tts(df[keys], df[dep], test_size = 0.3, random_state = 0)
    from sklearn.neighbors import KNeighborsClassifier
    knn_ = KNeighborsClassifier(n_neighbors=3)
    
    knn_.fit(A_train, b_train)
    b_pred=knn_.predict(A_test)
    filename = 'knn.sav'
    pickle.dump(knn, open(filename, 'wb'))
    return accuracy_score(b_test,b_pred), recall_score(b_test,b_pred)
 

def svm(df, dep, keys):
    from sklearn.model_selection import train_test_split as tts  
    A_train, A_test, b_train, b_test = tts(df[keys], df[dep], test_size = 0.3, random_state = 0)
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
 
    clf.fit(A_train, b_train)
    b_pred=clf.predict(A_test)
    filename = 'svm.sav'
    pickle.dump(clf, open(filename, 'wb'))
    return accuracy_score(b_test,b_pred), recall_score(b_test,b_pred)
 

def annModel(df, dep, keys):
    ann = Sequential()
    from sklearn.model_selection import train_test_split as tts  
    A_train, A_test, b_train, b_test = tts(df[keys], df[dep], test_size = 0.3, random_state = 0)
    
    #adding the input and first hidden layer
    ann.add(Dense(16, activation='relu', kernel_initializer='glorot_uniform',input_dim=len(keys)))
    #adding second layer
    ann.add(Dense(6, activation='relu', kernel_initializer='glorot_uniform'))
    #adding the output layer
    ann.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    ann.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    ann.fit(A_train, b_train,
        batch_size=35,
        epochs=50,
        shuffle=True,
    )

    b_pred=ann.predict(A_test)
    for i in range(len(b_pred)):
        if b_pred[i] >= 0.5:
            b_pred[i] = 1
        else:
            b_pred[i] = 0

    ann.save('ann.h5')
    return accuracy_score(b_test,b_pred), recall_score(b_test,b_pred)
    
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
    options = st.multiselect(
        'Select categorical variables if any',
        l
    )

    targets = st.multiselect(
        'Select target variables if any',
        l
    )

    failure = st.selectbox('Select failure field',
        l
    )

    visualize_l = st.selectbox('Select variable to visualize against failure',
        l
    )

    number = st.number_input('Enter value to replace Nan : ')

    return options,targets,failure,number,visualize_l


st.markdown("# Analytics")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    options,targets,failure,number,visualize_l=selectCatVar(df)

    df['datetime_new'] = pd.to_datetime(
        df['datetime'],
        format="%Y-%m-%d %H:%M:%S"
    )

    for option in options:
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

    if summary:
        st.write(df.describe())
    
    if visualize:
        
        fig, ax = plt.subplots()
        fig.set_figheight(6)
        fig.set_figwidth(10)
        dates = matplotlib.dates.date2num(df['datetime_new'])
        
        plt.plot_date(dates, df[visualize_l], linestyle='-', marker='', label='Voltage')

        # Plot Failures
        plt.plot_date(dates, df['failure'], color='red', linestyle='None', marker='o', markersize=5, label='Failures')

        # Title and labels
        plt.title("Failures with Voltage Variations")
        plt.xlabel("Date")
        plt.ylabel("Voltage")
    
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
        
