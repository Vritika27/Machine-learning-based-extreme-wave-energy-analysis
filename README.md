#Data loading & preprocessing
import pandas as pd

data_source = 'https://www.ndbc.noaa.gov/data/realtime2/46022.txt'
df = pd.read_csv(
    data_source,
    sep=r'\s+', # Use raw string 'r' to avoid SyntaxWarning
    skiprows=[1], # Skip the second header row that contains units
    na_values=['MM', 99.0, 999.0, 9999.0])
for c in ['WVHT','DPD','APD']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')#for data cleaning/preprocessing
      

#Wave power calculation formula
rho = 1025  # seawater density (kg/m³)
g = 9.81
df['Wave_Power'] = (rho * g**2 / (64 * np.pi)) * (df['WVHT']**2) * df[Tcol]

#ML model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = df[["WVHT", "DPD", "APD"]]
y = df["Wave_Power"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

#Performance evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
y_pred = model.predict(X_test)
y_test = np.array(y_test).ravel()
y_pred = np.array(y_pred).ravel()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
