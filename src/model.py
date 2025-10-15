from data_preparation import load_data, save_data
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# use one hot encoder for categorical features
class OneHotEncoder:
    def __init__(self, df: pd.DataFrame, categorical_features: list):
        self.df = df
        self.categorical_features = categorical_features

    def encode(self) -> pd.DataFrame:
        '''Apply one-hot encoding to the specified categorical features.'''
        self.df = pd.get_dummies(self.df, columns=self.categorical_features, drop_first=True)
        return self.df
    
# the dnn model
def build_dnn_model(input_shape: int) -> None:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_shape,))
    x = tf.keras.layers.Dense(128, activation='relu',name='l1',kernel_regularizer=tf.keras.regularizers.l2(0.008))(inputs)
    x = tf.keras.layers.Dropout(0.3,name='l1_dropout')(x)
    x = tf.keras.layers.Dense(32, activation='relu',name='l2',kernel_regularizer=tf.keras.regularizers.l2(0.008))(x)
    x = tf.keras.layers.Dropout(0.2,name='l2_dropout')(x)
    x = tf.keras.layers.Dense(16, activation='relu',name='l3',kernel_regularizer=tf.keras.regularizers.l2(0.008))(x)
    x = tf.keras.layers.Dropout(0.1,name='l3_dropout')(x)
    x = tf.keras.layers.Dense(8, activation='relu',name='l4')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid',name='output')(x)

    model = tf.keras.models.Model(inputs, outputs, name="DNN_Model")
    return model

# train model
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32) -> None:
    from tensorflow.keras import callbacks  # type: ignore

    # Define LR scheduler (reduce when plateau)
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.7, patience=5,
        min_lr=1e-6, verbose=1
    )
    # Define Early Stopping
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=35, restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, lr_scheduler]
    )
    return history

# plot training and validation accuracy and loss
def plot_training_history(history) -> None:
    import matplotlib.pyplot as plt

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Define the path to the data 
    train_data_path = os.path.join('src', 'data', 'feature_engineered_train.csv')
    test_data_path = os.path.join('src', 'data', 'feature_engineered_test.csv')
    # Load the data
    try:
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
    # One Hot Encoding
    categorical_features = ['Sex', 'Embarked', 'Title', 'Cabin_deck']
    if not train_data.empty and not test_data.empty:
        encoder = OneHotEncoder(train_data, categorical_features)
        train_data_encoded = encoder.encode()
        train_data_encoded = train_data_encoded.drop(columns=["PassengerId"])
        encoder = OneHotEncoder(test_data, categorical_features)
        test_data_encoded = encoder.encode()
        test_passenger_ids = test_data_encoded["PassengerId"]
        test_data_encoded = test_data_encoded.drop(columns=["PassengerId"])
        # Save the encoded data
        save_data(train_data_encoded, os.path.join('src', 'data', 'encoded_train.csv'))
        save_data(test_data_encoded, os.path.join('src', 'data', 'encoded_test.csv'))
    else:
        print("No data loaded.")
    
    # split the data into train validation sets and scale the numerical features
    X = train_data_encoded.drop(columns=["Survived"])
    y = train_data_encoded["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    num_features = ['Age', 'Fare', 'FamilySize']

    X_train[num_features] = scaler.fit_transform(X_train[num_features])
    X_val[num_features] = scaler.transform(X_val[num_features])
    # apply the same scaling to the test data
    test_data_encoded[num_features] = scaler.transform(test_data_encoded[num_features])
    # align the columns of test data to train data
    test_data_encoded = test_data_encoded.reindex(columns=X_train.columns, fill_value=0)
    # save the scaled data
    save_data(X_train, os.path.join('src','data', 'X_train.csv'))
    save_data(X_val, os.path.join('src','data', 'X_val.csv'))
    save_data(y_train, os.path.join('src','data', 'y_train.csv'))
    save_data(y_val, os.path.join('src','data', 'y_val.csv'))
    save_data(test_data_encoded, os.path.join('src','data', 'test_data.csv'))
    # let user decide load model or retrain R for retrain and L for load
    choice = input("Enter 'R' to retrain or 'L' to load a saved model: ").strip().upper() #TODO remove this and to the following
    # check if the model saved is same as the current model
    # if not same, retrain the model
    # else load the model
    if choice not in ('R', 'L'):
        print("Invalid choice, defaulting to retrain ('R').")
        choice = 'R'
    if choice == 'R':
        model = build_dnn_model(X_train.shape[1])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
        # Save the trained model
        model.save(os.path.join('src','models', 'dnn_model.h5'))
    elif choice == 'L':
        from tensorflow.keras.models import load_model  # type: ignore
        try:
            model = load_model(os.path.join('src','models', 'dnn_model.h5'))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = build_dnn_model(X_train.shape[1])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()
            history = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
            # Save the trained model
            model.save(os.path.join('src','models', 'dnn_model.keras'))

    # plot training and validation accuracy and loss
    if choice == 'R':
        plot_training_history(history)
    # evaluate the model on the validation set
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_acc:.4f}")
    # Make predictions on the test set
    predictions = model.predict(test_data_encoded)
    predictions = (predictions > 0.5).astype(int).flatten()
    # Prepare the submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': predictions
    })
    # Save the submission file
    save_data(submission, os.path.join('src','data', 'submissions', 'submission.csv'))
