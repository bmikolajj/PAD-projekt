import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
print("\nDataset")
df = pd.read_csv('weatherAUS.csv')

print(df.head())
df.info()

# Categorical variables overview
print("\nCategorical variables")
categorical_vars = df.select_dtypes(include=['object'])
print(categorical_vars)

nulls_in_categorical_vars = categorical_vars.isnull().sum()

print('Nulls in categorical variables counts: ')
print(nulls_in_categorical_vars)

for column in categorical_vars:
    unique_values = df[column].unique()
    print(f"\nUnique values for {column}: {unique_values}")

rain_today_counts = df['RainToday'].value_counts(dropna=False)
rain_tomorrow_counts = df['RainTomorrow'].value_counts(dropna=False)

print(f"\nCounts for RainToday: \n{rain_today_counts}")
print(f"\nCounts for RainTomorrow: \n{rain_tomorrow_counts}")

# Numerical variables overview
print("\nNumerical variables")
numerical_vars = df.select_dtypes(include=['int64', 'float64'])

nulls_in_numerical_vars = numerical_vars.isnull().sum()

print('\nNulls in numerical variables counts: ')
print(nulls_in_numerical_vars)

summary_statistics = numerical_vars.describe().round(2)
print("\nSummary statistics: ")
print(summary_statistics)

# Data cleanup
print("\nDate extraction")
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df = df.drop('Date', axis=1)

print("\nNumerical variables cleanup")
outlying_numerical_vars = ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']

fig = make_subplots(rows=2, cols=2)

for i, column in enumerate(outlying_numerical_vars, start=1):
    fig.add_trace(
        go.Box(y=df[column], name=column),
        row=(i-1)//2 + 1,
        col=(i-1)%2 + 1
    )

print("\nOutlying numerical variables: ")
fig.show()
    
outlying_numerical_vars = ['Rainfall', 'Evaporation', 'WindSpeed9am', 'WindSpeed3pm']

for column in outlying_numerical_vars:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Analyze data and its correlation
print("\nData analysis")
numerical_df = df.select_dtypes(include=[np.number])

correlation_matrix = numerical_df.corr()
print("\nCorrelation matrix: ")
print(correlation_matrix)

sorted_correlation = correlation_matrix['Rainfall'].sort_values(ascending=False)
fig_correlation_with_rainfall = go.Figure(data=go.Bar(x=sorted_correlation.index, y=sorted_correlation.values))
fig_correlation_with_rainfall.update_layout(title='Correlation with Rainfall', xaxis_title='Variables', yaxis_title='Correlation')
print("\nCorrelation with Rainfall only: ")
fig_correlation_with_rainfall.show()

heatmap = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=list(correlation_matrix.columns),
    y=list(correlation_matrix.index),
    annotation_text=correlation_matrix.round(2).values,
    showscale=True)

heatmap.show()

# Create dashboard
df['AvgTemp'] = (df['MinTemp'] + df['MaxTemp']) / 2
df_daily = df.groupby(['Year', 'Month', 'Day'])['AvgTemp'].mean().reset_index()
df_daily['Date'] = pd.to_datetime(df_daily[['Year', 'Month', 'Day']])
fig_avgtemp_by_date = go.Figure(data=go.Scatter(x=df_daily['Date'], y=df_daily['AvgTemp'], name='Daily avg temperature (Celsius)'))
fig_avgtemp_by_date.update_xaxes(title_text='Date')
fig_avgtemp_by_date.update_yaxes(title_text='Average Temperature (Celsius)')
fig_avgtemp_by_date.show()

rainfall_by_location = df.groupby('Location')['Rainfall'].sum()
fig_rainfall_by_location = go.Figure(data=go.Bar(x=rainfall_by_location.index, y=rainfall_by_location.values, name='Rainfall (mm)'))
fig_rainfall_by_location.update_xaxes(title_text='Location')
fig_rainfall_by_location.update_yaxes(title_text='Total Rainfall (mm)')
fig_rainfall_by_location.show()

fig_rain_today = px.pie(df, names='RainToday', title='Rain Today')
fig_rain_today.show()

fig_humidity_histogram = go.Figure(data=go.Histogram(x=df['Humidity3pm'], name='Humidity'))
fig_humidity_histogram.update_xaxes(title_text='Humidity at 3pm')
fig_humidity_histogram.update_yaxes(title_text='Count')
fig_humidity_histogram.show()

fig_pressure_vs_humidity = go.Figure(data=go.Scatter(x=df['Pressure3pm'], y=df['Humidity3pm'], mode='markers', name='Pressure vs Humidity'))
fig_pressure_vs_humidity.update_xaxes(title_text='Pressure at 3pm')
fig_pressure_vs_humidity.update_yaxes(title_text='Humidity at 3pm')
fig_pressure_vs_humidity.show()

fig_maxtemp_vs_raintoday = px.box(df, x='RainToday', y='MaxTemp', title='MaxTemp vs RainToday')
fig_maxtemp_vs_raintoday.show()

df_copy = df.copy()
df_copy['RainyDay'] = df_copy['Rainfall'].apply(lambda x: 'Yes' if x > 0 else 'No')

fig_rainy_day = px.pie(df_copy, names='RainyDay', title='Rainy Day')
fig_rainy_day.show()

fig_maxtemp_vs_mintemp = px.scatter(df_copy, x='MinTemp', y='MaxTemp', color='RainyDay', title='MaxTemp vs MinTemp')
fig_maxtemp_vs_mintemp.show()

# Logistic Regression model
label_encoder = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: label_encoder.fit_transform(col))

df = df.fillna(df.mean()) 
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))  

X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy is: ', accuracy)