 #Load CIFAR-10 Dataset
def load_cifar10():
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test, (32, 32, 3), 10

# Load MNIST Dataset
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, y_train, x_test, y_test, (28, 28, 1), 10

# Load Tiny ImageNet (Example)
def load_tiny_imagenet():
    # Tiny ImageNet download and preprocessing should be completed before using this function
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_dir = "./tiny-imagenet-200/train"
    val_dir = "./tiny-imagenet-200/val"

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_data = datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode="categorical")
    val_data = datagen.flow_from_directory(val_dir, target_size=(64, 64), batch_size=32, class_mode="categorical")

    x_train, y_train = next(iter(train_data))
    x_test, y_test = next(iter(val_data))
    return x_train, y_train, x_test, y_test, (64, 64, 3), 200
#