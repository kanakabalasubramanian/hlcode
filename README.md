# hlcode
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# Define the data (City names and Period precipitation)
data = {
    'City name': ['Abiko City', 'Katori City', 'Tosho Town, Katori District', 'Funabashi City', 'Sakura City',
                  'Narita City', 'Choshi City', 'Yokoshibahikari-cho, Sammu-gun', 'Chuo Ward, Chiba City',
                  'Mobara City', 'Kisarazu City', 'Ichihara City', 'Kimitsu City', 'Otaki Town, Isumi District',
                  'Kyonan Town, Awa District', 'Kamogawa City', 'Katsuura City', 'Tateyama City'],
    'Period precipitation': [112.0, 144.5, 168.0, 117.5, 137.0, 151.5, 112.0, 132.5, 109.0, 146.0, 145.5,
                             221.5, 237.5, 212.0, 211.0, 140.5, 99.5, 192.0],
    'Flood': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1]
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Label encode the City names
label_encoder = LabelEncoder()
df['City label'] = label_encoder.fit_transform(df['City name'])

# Split the data into features (X) and target variable (y)
X = df[['City label', 'Period precipitation']]
y = df['Flood']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Create and train the ANN model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

print(predictions)
