"""
Implements the Random Forest model from the notebook
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from pathlib import Path

class F1Model:
    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)
        self.preprocessor = None

    def train(self, X, y):
        """Replicates the training process from the notebook"""
        from ml_logic.preprocessor import F1Preprocessor

        preprocessor = F1Preprocessor()
        processed_data = preprocessor.prepare_data(X)
        self.preprocessor = preprocessor.build_preprocessor()

        X_processed = self.preprocessor.fit_transform(processed_data)
        self.model.fit(X_processed, y)

        # Cross-validation (optional)
        # from sklearn.model_selection import cross_val_score
        # scores = cross_val_score(self.model, X_processed, y, cv=5, scoring='r2')
        # print(f"Cross-validation R²: {np.mean(scores):.3f} (±{np.std(scores):.3f})")

    def predict(self, X):
        """Make predictions"""
        processed_data = self.preprocessor.transform(X)
        return self.model.predict(processed_data)

    def evaluate(self, X_test, y_test):
        """Full evaluation as in notebook"""
        y_pred = self.predict(X_test)
        return {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }

    def save(self, path):
        """Save model artifacts"""
        joblib.dump({
            'model': self.model,
            'preprocessor': self.preprocessor
        }, path)

    @classmethod
    def load(cls, path):
        """Load model artifacts"""
        artifacts = joblib.load(path)
        model = cls()
        model.model = artifacts['model']
        model.preprocessor = artifacts['preprocessor']
        return model
