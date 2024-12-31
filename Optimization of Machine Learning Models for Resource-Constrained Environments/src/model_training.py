# Define the Lightweight Model
def create_lightweight_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), strides=(2, 2), padding="same", input_shape=input_shape),
        BatchNormalization(),
        ReLU(),

        DepthwiseConv2D((3, 3), padding="same"),
        BatchNormalization(),
        ReLU(),
        Conv2D(64, (1, 1), padding="same"),
        BatchNormalization(),
        ReLU(),

        GlobalAveragePooling2D(),
        Dense(num_classes, activation="softmax")
    ])
    return model