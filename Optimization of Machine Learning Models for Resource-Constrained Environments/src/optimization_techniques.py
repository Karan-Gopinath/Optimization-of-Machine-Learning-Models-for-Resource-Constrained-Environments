# Evaluate Latency of Uncompressed Model
def evaluate_latency(model, sample_input):
    start_time = time.time()
    model.predict(sample_input)
    end_time = time.time()
    return end_time - start_time

# Evaluate Latency of Compressed Model
def evaluate_tflite_latency(tflite_path, sample_input):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = sample_input.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
    return end_time - start_time