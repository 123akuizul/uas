import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Izul Ramdani ")
st.write("##### Nim   : 200411100111 ")
st.write("##### Kelas : Penambangan Data A ")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

df = pd.read_csv('https://raw.githubusercontent.com/123akuizul/uas/main/milknew.csv')

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Milk Quality Prediction (Prediksi Kualitas Susu) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/cpluzshrijayan/milkquality ")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. Kolom Ph ini mendefinisikan PH alus susu yang berkisar antara 3 hingga 9,5 maks : 6,25 hingga 6,90""")
    st.write("""2. Kolom Suhu ini mendefinisikan Suhu susu yang berkisar dari 34'C hingga 90'C maks : 34'C hingga 45,20'C""")
    st.write("""3. Kolom Rasa Susu ini mendefinisikan Rasa susu yang merupakan data kategori 0 (Buruk) atau 1 (Baik) maks : 1 (Baik)""")
    st.write("""4. Kolom Bau Susu ini mendefinisikan Bau susu yang merupakan data kategori 0 (Buruk) atau 1 (Baik) maks : 0 (Buruk)
    5. Kolom Kekeruhan Susu ini mendefinisikan Kekeruhan susu yang merupakan data kategorikal 0 (Rendah) atau 1 (Tinggi) maks : 1
    6. Kolom Warna Susu ini menentukan Warna susu yang berkisar dari 240 hingga 255 maks : 255
    7. Kolom Target ini mendefinisikan Grade (Target) susu yang merupakan data kategori Dimana Rendah (Buruk) atau Sedang (Sedang)""")
    st.write("###### Aplikasi ini untuk : Milk Quality Prediction (Prediksi Kualitas Susu) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : ")

with upload_data:
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)
    # df = pd.read_csv('')
    # st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['Grade'])
    y = df['Grade'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.Grade).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]]
    })

    st.write(labels)

with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        pH = st.number_input('Masukkan pH : ')
        Temprature = st.number_input('Masukkan Temprature : ')
        Taste = st.number_input('Masukkan Taste : ')
        Odor = st.number_input('Masukkan Odor : ')
        Fat = st.number_input('Masukkan Fat : ')
        Turbidity = st.number_input('Masukkan Turbidity : ')
        Colour = st.number_input('Masukkan Colour : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                pH,
                Temprature,
                Taste,
                Odor,
                Fat,
                Turbidity,
                Colour
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
