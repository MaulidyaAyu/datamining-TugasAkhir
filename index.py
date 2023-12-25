import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Fungsi untuk menganalisis pola pembelian pelanggan tertentu
def analyze_customer_purchase_patterns(data, customer_name):
    customer_data = data[data['pelanggan'] == customer_name]
    if not customer_data.empty:
        encoded_data = pd.get_dummies(customer_data.set_index('invoice')['deskripsi']).groupby(level='invoice').max()
        frequent_itemsets = apriori(encoded_data, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
        top_10_rules = rules.head(10)
        top_10_rules['antecedents'] = top_10_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
        top_10_rules['consequents'] = top_10_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))
        return top_10_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    else:
        return None


st.title('Analisis Pola Pembelian')
uploaded_file = st.file_uploader("Unggah file Excel", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.sidebar.subheader('Pilihan Fitur')
    selected_feature = st.sidebar.radio('', ('Visualisasi Data', 'Analisis Pelanggan'))

    if selected_feature == 'Visualisasi Data':
        selected_feature = st.sidebar.selectbox('Pilihan Visualisasi', ('Informasi Dataset', 'Item yang Sering Dibeli', 'Pelanggan yang Sering Membeli', 'Total Pembelian Tiap Bulan'))
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if selected_feature == 'Informasi Dataset':
            st.title('Informasi Dataset')
            st.write(f"Jumlah Record Data: {len(data)}")
            st.write('Preview Dataset')
            st.write(data.head())
            st.write('Daftar Nama Produk dalam Dataset:')
            unique_items = data['deskripsi'].unique()
            for idx, item in enumerate(unique_items, start=1):
                st.write(f"{idx}. {item}")

        elif selected_feature == 'Item yang Sering Dibeli':
            st.title('Item yang Sering Dibeli')
            top_items = data['deskripsi'].value_counts().head(10)
            plt.figure(figsize=(6, 4))
            ax = sns.barplot(x=top_items.values, y=top_items.index, palette='pink') 
            ax.set_xlabel('Jumlah Pembelian') 
            ax.set_ylabel('Item') 
            st.pyplot()

        elif selected_feature == 'Pelanggan yang Sering Membeli':
            st.title('Pelanggan yang Sering Membeli')
            top_customers = data['pelanggan'].value_counts().head(10)
            plt.figure(figsize=(6, 4))
            ax = sns.barplot(x=top_customers.values, y=top_customers.index, palette='pink')  
            ax.set_xlabel('Jumlah Pembelian')  
            ax.set_ylabel('Pelanggan')
            st.pyplot()

        elif selected_feature == 'Total Pembelian Tiap Bulan':
            st.title('Banyaknya Pembelian di Tiap Bulan')
            purchase_by_month = data.groupby(data['tanggal'].dt.to_period('M')).size()
            plt.figure(figsize=(8, 6))
            ax = purchase_by_month.plot(kind='bar', color='pink')  
            ax.set_xlabel('Bulan')  
            ax.set_ylabel('Jumlah Pembelian')  
            st.pyplot()

    elif selected_feature == 'Analisis Pelanggan':
        minimal_pembelian = st.number_input('Masukkan minimal pembelian:', min_value=0)
        filtered_customers = data.groupby('pelanggan').size()[lambda x: x >= minimal_pembelian].index.tolist()
        selected_customer = st.selectbox('Pilih Pelanggan:', filtered_customers)
        unique_dates = pd.to_datetime(data['tanggal'], errors='coerce').dt.to_period('M').sort_values().unique()
        start_month = st.selectbox('Pilih Bulan Awal:', options=unique_dates)
        end_month = st.selectbox('Pilih Bulan Akhir:', options=unique_dates, index=len(unique_dates) - 1)
        min_support = st.slider('Minimal Support', 0.01, 0.11, 0.05, step=0.01)
        min_lift = st.slider('Minimal Lift', 1.0, 7.0, 1.0, step=0.1)
        min_confidence = st.slider('Minimal Confidence', 0.1, 1.0, 0.5, step=0.1)

        if st.button('Analisis'):
            filtered_data = data[(data['pelanggan'] == selected_customer) & (pd.to_datetime(data['tanggal'], errors='coerce').dt.to_period('M').between(start_month, end_month))]
            hasil_pola_pembelian = analyze_customer_purchase_patterns(data, selected_customer)
            if not filtered_data.empty:
                encoded_data = pd.get_dummies(filtered_data.set_index('invoice')['deskripsi']).groupby(level='invoice').max()
                frequent_itemsets = apriori(encoded_data, min_support=min_support, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
                rules = rules[(rules['confidence'] >= min_confidence)]
                rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
                if rules is not None and not rules.empty:
                    st.subheader(f'Aturan Asosiasi untuk {selected_customer} dari {start_month} hingga {end_month}:')
                    for index, row in rules.head(5).iterrows():
                        antecedents = row['antecedents']
                        consequents = row['consequents']
                        confidence = row['confidence']
                        support = row['support']

                        antecedent_names = ', '.join([str(item) for item in antecedents])
                        consequent_names = ', '.join([str(item) for item in consequents])

                        st.write(f"Jika beli {antecedent_names}, maka beli {consequent_names} dengan confidence: {confidence}")
                else:
                    st.write(f"Tidak ada data pola pembelian untuk pelanggan {selected_customer} pada rentang bulan {start_month} sampai {end_month}.")
            if hasil_pola_pembelian is not None:
                    st.subheader(f'5 Item teratas yang sering dibeli oleh {selected_customer}:')
                    top_5_items = filtered_data['deskripsi'].value_counts().head(5)
                    st.write(top_5_items)
                    st.subheader(f"10 Aturan teratas pola pembelian untuk pelanggan {selected_customer}:")
                    st.table(hasil_pola_pembelian)

    # elif selected_feature == 'Tata Letak Produk':
    #     # ... Bagian untuk Tata Letak barang jika memungkinkan ...