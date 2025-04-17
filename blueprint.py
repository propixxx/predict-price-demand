import json
import pandas as pd
import numpy as np
import joblib
import os
import logging
from geopy.distance import geodesic
import azure.functions as func
from xgboost import XGBRegressor

# Inisialisasi objek model

# Load dari file yang telah disimpan

# Load models & preprocessors
scaler = joblib.load(os.path.join(os.getcwd(), "scaler_Kmeans.pkl"))
kmeans = joblib.load(os.path.join(os.getcwd(), "kmeans_model.pkl"))
preprocessor = joblib.load(os.path.join(os.getcwd(), "preprocessor.pkl"))
model = XGBRegressor()
model.load_model("best_xgb_model.model")  # atau .json / .bin

# Dataset
df_merge = pd.read_csv("https://raw.githubusercontent.com/propixxx/price_predict/refs/heads/main/DATA_PROYEKSI.csv", sep=';')
df_merge.drop(columns=['Status Pro'], inplace=True)
df_merge[['Latitude', 'Longitude']] = df_merge['LatLong'].str.split(',', expand=True)
df_merge['Latitude'] = df_merge['Latitude'].astype(float)
df_merge['Longitude'] = df_merge['Longitude'].astype(float)
df_merge.drop(columns=['LatLong'], inplace=True)

# KMeans prediction pada dataset awal
df_kmeans_old = df_merge[['Longitude', 'Latitude']]
df_kmeans_scaled = scaler.transform(df_kmeans_old)
df_predict_old = pd.DataFrame(df_kmeans_scaled, columns=['Longitude', 'Latitude'])
df_predict_old['Cluster'] = kmeans.predict(df_kmeans_scaled)
df_predict_old[['Longitude', 'Latitude']] = scaler.inverse_transform(df_predict_old[['Longitude', 'Latitude']])

# Define passthrough features
passthrough_features = [
    'Bandar Udara', 'Kebun/Perkebunan', 'Permukiman Perdesaan',
    'Permukiman Perkotaan', 'Permukiman Semi Desa', 'Rumput/Tanah Kosong',
    'Sawah irigasi', 'Tegalan/Ladang'
]

def normalize_columns(cols):
    return cols.str.replace(' ', '').str.replace('/', '')

def get_columns_by_year(columns, features, year):
    normalized_cols = normalize_columns(columns)
    normalized_feats = [f.replace(' ', '').replace('/', '') for f in features]
    mapping = dict(zip(normalized_cols, columns))
    return [mapping[col] for col in normalized_cols if any(f in col for f in normalized_feats) and str(year) in col]

def convert_percentage_columns(df, columns):
    for col in columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].str.rstrip('%').str.replace(',', '.', regex=False), errors='coerce').fillna(0)
    return df

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="propix_predict_price", auth_level=func.AuthLevel.FUNCTION)
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('Python HTTP trigger function processed a request.')
        logging.info(f"Request body: {req.get_body()}")
        logging.info(f"Request query: {req.params}")
        input_data = req.get_json()
        lat, lon = input_data['Latitude'], input_data['Longitude']
        current_location = (lat, lon)

        # KMeans cluster dari properti baru
        df_input_Kmeans = pd.DataFrame({'Longitude': [lon], 'Latitude': [lat]})
        cluster = kmeans.predict(df_input_Kmeans)[0]

        df_cluster = df_predict_old[df_predict_old['Cluster'] == cluster]
        df_cluster['distance'] = df_cluster.apply(lambda row: geodesic(current_location, (row['Latitude'], row['Longitude'])).km, axis=1)
        nearest = df_cluster.nsmallest(3, 'distance')
        df_nearest = df_merge.loc[nearest.index]

        # Ganti kategori ordinal
        for feat in ['EKON', 'NTL', 'KEP']:
            for year in [2025, 2030, 2035]:
                col = f"{feat}{year}"
                df_nearest[col] = df_nearest[col].replace({'Rendah': 1, 'Sedang': 2, 'Tinggi': 3})

        # Pisahkan kolom berdasarkan tahun
        column_2025 = get_columns_by_year(df_nearest.columns, passthrough_features, 2025)
        column_2030 = get_columns_by_year(df_nearest.columns, passthrough_features, 2030)
        column_2035 = get_columns_by_year(df_nearest.columns, passthrough_features, 2035)

        X_2025 = convert_percentage_columns(df_nearest, column_2025)
        X_2030 = convert_percentage_columns(df_nearest, column_2030)
        X_2035 = convert_percentage_columns(df_nearest, column_2035)

        average_values = df_nearest[column_2025 + column_2030 + column_2035].mean()
        rounded_values = df_nearest[[f"{f}{y}" for f in ['NTL','KEP','EKON'] for y in [2025,2030,2035]]].mean().round().astype(int)
        inverse_mapping = {1: 'Rendah', 2: 'Sedang', 3: 'Tinggi'}
        mapped_values = rounded_values.replace(inverse_mapping)

        df_average = pd.DataFrame(average_values).T
        df_average[mapped_values.index] = mapped_values.values

        df_input = pd.DataFrame(input_data, index=[0])
        df_combined = pd.concat([df_input, df_average], axis=1)
        df_combined['Rasio Luas Bangunan'] = df_combined['Luas Bangu'] / df_combined['Luas Tanah']

        selected_col = ['Kategori P', 'Luas Bangu', 'Luas Tanah', 'Jenis Sert', 'Rasio Luas Bangunan', 'Latitude', 'Longitude', 'Dekat Exit Tol']

        def predict_for_year(year):
            features = selected_col + [col for col in df_merge.columns if str(year) in col]
            X = df_combined[features].copy()
            X.columns = X.columns.str.replace(str(year), '').str.rstrip()
            preprocessed = preprocessor.transform(X)
            pred = model.predict(preprocessed)
            return f"{np.expm1(pred[0]):.0f}"

        y_pred = {
            "Estimated Value 2025": predict_for_year(2025),
            "Predicted 2030": predict_for_year(2030),
            "Predicted 2035": predict_for_year(2035)
        }

        return func.HttpResponse(json.dumps(y_pred), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.exception("Prediction failed.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)

# import azure.functions as func
# import logging

# app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# @app.route(route="propix_predict_price")
# def propix_predict_price(req: func.HttpRequest) -> func.HttpResponse:
#     logging.info('Python HTTP trigger function processed a request.')

#     name = req.params.get('name')
#     if not name:
#         try:
#             req_body = req.get_json()
#         except ValueError:
#             pass
#         else:
#             name = req_body.get('name')

#     if name:
#         return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
#     else:
#         return func.HttpResponse(
#              "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
#              status_code=200
#         )
