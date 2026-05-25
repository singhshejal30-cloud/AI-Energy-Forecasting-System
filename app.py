# =========================================================
# INDUSTRY-LEVEL ENERGY CONSUMPTION FORECASTING SYSTEM
# Fast + Accurate + Error-Free Streamlit App
# =========================================================

# Run Command:
# streamlit run app.py

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Energy Forecasting",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# CUSTOM CSS + BACKGROUND
# =========================================================

page_bg = """
<style>

html, body, [class*="css"]  {
    color: white !important;
}

[data-testid="stAppViewContainer"]{
background-image: linear-gradient(
rgba(0,0,0,0.75),
rgba(0,0,0,0.75)),
url("https://images.unsplash.com/photo-1473341304170-971dccb5ac1e");

background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"]{
background-color: rgba(0,0,0,0.7);
}

/* All Text White */

h1,h2,h3,h4,h5,h6,p,label,span,div{
color:white !important;
}

/* Metrics */

.stMetric{
background-color: rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
color:white !important;
}

/* Dropdown Box */

.stSelectbox div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}

/* Dropdown Options */

div[role="listbox"] ul {
    background-color: white !important;
}

div[role="option"] {
    color: black !important;
    background-color: white !important;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("⚡ AI-Powered Energy Consumption Forecasting")

st.markdown(
"""
<h3 style='text-align:center;color:white;'>
Industry-Level Deep Learning Forecasting Dashboard
</h3>
""",
unsafe_allow_html=True
)

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("AEP_hourly.csv")

    df.columns = ['Datetime', 'Energy']

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df = df.sort_values('Datetime')

    df.set_index('Datetime', inplace=True)

    return df

df = load_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['day_of_week'] = df.index.dayofweek
df['year'] = df.index.year

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

sequence_length = st.sidebar.slider(
    "Sequence Length",
    12,
    72,
    24
)

# =========================================================
# DATA PREVIEW
# =========================================================

st.subheader("📂 Dataset Preview")

st.dataframe(df.head())

# =========================================================
# ENERGY GRAPH
# =========================================================

st.subheader("📈 Energy Consumption Analysis")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Energy'],
    mode='lines',
    name='Energy Consumption'
))

fig.update_layout(
    template='plotly_dark',
    height=500,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title='Date',
    yaxis_title='Energy'
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# SCALING
# =========================================================

data = df[['Energy']]

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# =========================================================
# CREATE SEQUENCES
# =========================================================

X = []
y = []

for i in range(sequence_length, len(scaled_data)):

    X.append(
        scaled_data[i-sequence_length:i, 0]
    )

    y.append(
        scaled_data[i,0]
    )

X = np.array(X)
y = np.array(y)

# =========================================================
# RESHAPE DATA
# =========================================================

X = np.reshape(
    X,
    (X.shape[0], X.shape[1], 1)
)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# =========================================================
# LOAD OR CREATE MODEL
# =========================================================

st.subheader("🧠 AI Model")

try:

    model = load_model("energy_forecasting_model.h5")

    st.success("✅ Pretrained Model Loaded Successfully!")

except:

    st.warning("⚠️ No Saved Model Found. Training New Model...")

    model = Sequential()

    model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(X_train.shape[1],1)
        )
    )

    model.add(Dropout(0.2))

    model.add(LSTM(32))

    model.add(Dropout(0.2))

    model.add(Dense(16))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_test,y_test),
        verbose=1
    )

    model.save("energy_forecasting_model.h5")

    st.success("✅ Model Trained & Saved Successfully!")

# =========================================================
# PREDICTIONS
# =========================================================

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1,1)
)

# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test_actual,
    predictions
)

mse = mean_squared_error(
    y_test_actual,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test_actual,
    predictions
)

# =========================================================
# PERFORMANCE METRICS
# =========================================================

st.subheader("📌 Model Performance")

col1,col2,col3,col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R² Score", f"{r2:.4f}")

# =========================================================
# ACTUAL VS PREDICTED GRAPH
# =========================================================

st.subheader("🔮 Actual vs Predicted Energy")

pred_fig = go.Figure()

pred_fig.add_trace(go.Scatter(
    y=y_test_actual.flatten(),
    mode='lines',
    name='Actual Energy'
))

pred_fig.add_trace(go.Scatter(
    y=predictions.flatten(),
    mode='lines',
    name='Predicted Energy'
))

pred_fig.update_layout(
    template='plotly_dark',
    height=550,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

st.plotly_chart(pred_fig, use_container_width=True)

# =========================================================
# FUTURE PREDICTION
# =========================================================

st.subheader("⚡ Next Hour Energy Prediction")

future = model.predict(
    X_test[-1].reshape(1,sequence_length,1)
)

future = scaler.inverse_transform(future)

st.metric(
    "Predicted Future Energy",
    f"{future[0][0]:.2f}"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
"""
<h3 style='text-align:center;color:white;'>
🚀 Developed with Streamlit + Deep Learning + AI Forecasting
</h3>
""",
unsafe_allow_html=True
)