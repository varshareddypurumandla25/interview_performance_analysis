# model.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class InterviewPerformanceModel:

    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, df):
        X = df.drop('Performance', axis=1)
        y = df['Performance']

        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Train models
        self.kmeans.fit(X_scaled)
        self.rf_model.fit(X_scaled, y)

    def predict_performance(self, input_data):
        input_scaled = self.scaler.transform([input_data])
        prediction = self.rf_model.predict(input_scaled)[0]
        cluster = self.kmeans.predict(input_scaled)[0]

        return prediction, cluster