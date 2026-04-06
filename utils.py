# utils.py

import pandas as pd
import numpy as np

def generate_sample_data(n=100):
    data = {
        "Aptitude": np.random.randint(40, 100, n),
        "Communication": np.random.randint(40, 100, n),
        "Technical": np.random.randint(40, 100, n),
        "Confidence": np.random.randint(40, 100, n),
        "Performance": np.random.choice(["Poor", "Average", "Good"], n)
    }
    return pd.DataFrame(data)


def get_feedback(prediction):
    if prediction == "Good":
        return "Excellent performance! Keep it up 👍"
    elif prediction == "Average":
        return "You can improve with more practice."
    else:
        return "Needs improvement. Focus on fundamentals."