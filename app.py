import datetime
import streamlit as st
import pandas as pd
from google.cloud import storage
import io
from dateutil import parser as date_parser
from BFB_CCC_logic import main as BFB_logic_main
import base64

import time
start_time = time.time()


KEY_FILE_PATH = "C:/Users/Kavya/kavVSCode/gcs_keys/symmetric-hash-413516-250618204f5d.json"

required_columns = ['Product Name', 'Brand Name', 'Price', 'Quantity/Size', 'Shade/Color', 'Ingredients', 'Product Type', 'Usage Instructions', 'Expiration Date', 'Manufacturing Date', 'Country of Origin', 'Special Features', 'Certifications', 'Description', 'Image Path', 'Benefits', 'Skin Type Compatibility', 'Packaging', 'Product Images', 'Return Policy', 'Disclaimer']

def read_product_data(filename):
    df = pd.DataFrame()

    try:
        client = storage.Client.from_service_account_json(KEY_FILE_PATH)
        bucket = client.get_bucket("catscore-bucket01")
        blob = bucket.blob(filename)

        with io.BytesIO() as bio:
            blob.download_to_file(bio)
            bio.seek(0)
            df = pd.read_csv(bio)

        st.write(df.head())
    except Exception as e:
        st.error(f"An error occurred while reading the file: {str(e)}")

    return df

def check_columns(df):
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns

def main():
    st.title("Score Your Catalogue:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    prod_data_df = pd.DataFrame()

    if uploaded_file is not None:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

        missing_columns = check_columns(df)
        if missing_columns:
            st.error(f"The following columns are missing in the uploaded file: {', '.join(missing_columns)}")
        else:
            # bucket_name = 'catscore-bucket01'
            client = storage.Client.from_service_account_json(KEY_FILE_PATH)
            export_bucket = client.get_bucket('catscore-bucket01')
            df.to_csv()
            filename = uploaded_file.name +'-{0}'.format(datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
            export_bucket.blob(filename).upload_from_string(df.to_csv(), 'text/csv')

            st.success(f"Success message: CSV file saved into Cloud Storage")

            prod_data_df = read_product_data(filename)

            res_scores = BFB_logic_main(prod_data_df)
            st.write(res_scores.head())
            export_bucket = client.get_bucket('catscore-bucket01')
            res_scores.to_csv()
            res_file_name = "scores-"+filename
            export_bucket.blob(res_file_name).upload_from_string(res_scores.to_csv(), 'text/csv')
            # st.write(res_scores.to_csv())

            csv = res_scores.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Convert DataFrame to bytes and encode as base64
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download Catalogue Scoring results</a>'
            
            st.markdown(href, unsafe_allow_html=True)

            st.write("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
