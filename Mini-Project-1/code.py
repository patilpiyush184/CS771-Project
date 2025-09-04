# Import necessary libraries for data manipulation, visualization, and model building.
import numpy as np  # For numerical operations and array handling
import pandas as pd  # For data manipulation and CSV reading
import matplotlib.pyplot as plt  # For data visualization (if needed for future plots)

# Import scikit-learn tools for preprocessing and model building
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Standardizing data and encoding categorical data
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, log_loss  # Evaluation metrics
from sklearn.ensemble import RandomForestClassifier  # Random Forest model for classification
from sklearn.linear_model import LogisticRegression  # Logistic regression for feature-based model

# Import TensorFlow/Keras tools for deep learning models
from tensorflow.keras.models import Sequential  # Sequential model for stacking layers
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, LSTM, Conv1D, GRU  # Neural network layers
from tensorflow.keras.preprocessing.sequence import pad_sequences  # For padding sequences to the same length
from tensorflow.keras.optimizers import Adam  # Adam optimizer for training the models

# Define the Text Sequence Model class to process textual data
class TextSeqModel():
    def __init__(self) -> None:
        """Initialize the Text Sequence Model by building its architecture."""
        self.model = self.build_model()

    def build_model(self):
        """Build the neural network for text sequences."""
        model = Sequential([
            # Embedding layer to convert text input into dense vector representations
            Embedding(input_dim=10, output_dim=8, input_length=50),
            
            # 1D Convolutional layer to capture patterns in sequences
            Conv1D(32, kernel_size=5, activation='relu'),
            
            # GRU layer to capture temporal dependencies in sequences
            GRU(32),
            
            # Fully connected layer with ReLU activation for non-linear transformation
            Dense(16, activation='relu'),
            
            # Output layer with sigmoid activation for binary classification
            Dense(1, activation='sigmoid')
        ])
        # Compile the model using binary cross-entropy loss and Adam optimizer
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        """Train the model using the provided training data."""
        self.model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    def predict(self, X):
        """Generate predictions for the given input data."""
        return (self.model.predict(X) > 0.5).astype(int)

# Define the Emoticon Model class to handle emoticon-based inputs
class EmoticonModel():
    def __init__(self, input_dim) -> None:
        """Initialize the Emoticon Model with a specific input dimension."""
        self.model = self.build_model(input_dim)

    def build_model(self, input_dim):
        """Build the neural network for emoticon sequences."""
        model = Sequential([
            # Embedding layer to convert emoticons into dense vectors
            Embedding(input_dim=input_dim, output_dim=8, input_length=13),
            
            # Simple RNN layer to capture sequential patterns
            SimpleRNN(16, return_sequences=True),
            
            # LSTM layer for capturing long-term dependencies
            LSTM(32),
            
            # Output layer with sigmoid activation for binary classification
            Dense(1, activation='sigmoid')
        ])
        # Compile the model using binary cross-entropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X, y):
        """Train the emoticon model with the given data."""
        self.model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    def predict(self, X):
        """Generate predictions based on input emoticon sequences."""
        return (self.model.predict(X) > 0.5).astype(int)

# Define the Feature Model class using Logistic Regression
class FeatureModel():
    def __init__(self) -> None:
        """Initialize the feature-based model with standard scaling."""
        self.scaler = StandardScaler()  # Standardize features for better performance
        self.model = SVC(probability=True)  # Use SVC for classification

    def train(self, X, y):
        """Train the feature model using scaled input data."""
        X_scaled = self.scaler.fit_transform(X)  # Standardize the features
        self.model.fit(X_scaled, y)  # Train the logistic regression model

    def predict(self, X):
        """Generate predictions after scaling the input data."""
        X_scaled = self.scaler.transform(X)  # Apply the same scaling to test data
        return self.model.predict(X_scaled)

# Define the Combined Model class using Random Forest
class CombinedModel():
    def __init__(self) -> None:
        """Initialize the combined model using Random Forest."""
        self.model = RandomForestClassifier()

    def train(self, X, y):
        """Train the combined model using concatenated features."""
        self.model.fit(X, y)

    def predict(self, X):
        """Generate predictions based on the combined input data."""
        return self.model.predict(X)

# Utility function to save predictions to a text file
def save_predictions_to_file(predictions, filename):
    """Save the generated predictions to a specified file."""
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

# Main script execution
if __name__ == '__main__':
    # Load and preprocess text sequence data
    train_seq_data = pd.read_csv("datasets/train/train_text_seq.csv")
    train_seq_X = pad_sequences(train_seq_data['input_str'].apply(lambda x: list(map(int, x))).tolist(), maxlen=50)
    train_seq_y = train_seq_data['label'].values

    text_model = TextSeqModel()  # Initialize text sequence model
    text_model.train(train_seq_X, train_seq_y)  # Train the model

    # Load and preprocess emoticon data
    train_emoticon_data = pd.read_csv("datasets/train/train_emoticon.csv")
    train_emoticon_data['emoticons'] = train_emoticon_data['input_emoticon'].apply(lambda x: list(x))
    all_emoticons = train_emoticon_data['emoticons'].explode().unique()
    emoticon_to_index = {emoticon: idx + 1 for idx, emoticon in enumerate(all_emoticons)}

    train_emoticon_X = pad_sequences(
        train_emoticon_data['emoticons'].apply(lambda lst: [emoticon_to_index.get(e, 0) for e in lst]).tolist(), maxlen=13
    )
    train_emoticon_y = train_emoticon_data['label'].values

    emoticon_model = EmoticonModel(input_dim=len(emoticon_to_index) + 1)
    emoticon_model.train(train_emoticon_X, train_emoticon_y)

    # Load and preprocess feature-based data
    feature_train_data = np.load("datasets/train/train_feature.npz", allow_pickle=True)
    train_feature_X = feature_train_data['features'].reshape(feature_train_data['features'].shape[0], -1)
    train_feature_y = feature_train_data['label']

    feature_model = FeatureModel()
    feature_model.train(train_feature_X, train_feature_y)

    # Combine all models' features and train the combined model
    combined_X = np.concatenate([train_feature_X, train_emoticon_X, train_seq_X], axis=1)
    combined_y = train_feature_y

    best_model = CombinedModel()
    best_model.train(combined_X, combined_y)

    # Load validation data and make predictions
    test_feature_X = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)['features'].reshape(-1, 9984)
    test_emoticon_X = pad_sequences(
        pd.read_csv("datasets/valid/valid_emoticon.csv")['input_emoticon'].apply(lambda x: [emoticon_to_index.get(e, 0) for e in list(x)]).tolist(), maxlen=13
    )
    test_seq_X = pad_sequences(
        pd.read_csv("datasets/valid/valid_text_seq.csv")['input_str'].apply(lambda x: list(map(int, x))).tolist(), maxlen=50
    )
    test_combined_X = np.concatenate([test_feature_X, test_emoticon_X, test_seq_X], axis=1)

    # Generate predictions and save to files
    pred_feat = feature_model.predict(test_feature_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_combined_X)

    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
