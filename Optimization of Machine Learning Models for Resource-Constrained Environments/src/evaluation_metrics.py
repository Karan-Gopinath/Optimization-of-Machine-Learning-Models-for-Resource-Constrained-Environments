 #Function to Train and Evaluate Model
def train_and_evaluate(dataset_name, x_train, y_train, x_test, y_test, input_shape, num_classes):
    print(f"Training on {dataset_name}...")
    model = create_lightweight_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32)

    # Save the uncompressed model
    model_path = f"/content/drive/MyDrive/{dataset_name}_model.h5"
    model.save(model_path)
    print(f"Model saved at {model_path}")

    # Compress the model using TensorFlow Lite
    print("Compressing model using TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_model_path = f"/content/drive/MyDrive/{dataset_name}_model.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Compressed model saved at {tflite_model_path}")

    # Evaluate both models
    uncompressed_acc = max(history.history['val_accuracy'])
    uncompressed_latency = evaluate_latency(model, x_test[:1])

    compressed_latency = evaluate_tflite_latency(tflite_model_path, x_test[:1])

    # Return metrics
    return {
        "Dataset": dataset_name,
        "Uncompressed Accuracy": uncompressed_acc,
        "Uncompressed Latency (s)": uncompressed_latency,
        "Compressed Latency (s)": compressed_latency,
        "Model Size (MB)": len(tflite_model) / 1e6
    }
