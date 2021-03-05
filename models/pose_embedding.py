from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import tensorflow as tf

class PreprocessVGG16(Model):
    def __init__(self, image_size):
        self.image_size = image_size
    def call(self, x, training=None, mask=None):
        x = vgg16.preprocess_input(x * 255)
        return tf.image.resize(x, self.image_size)

class PoseEmbeddings(Model):
        def __init__(self, image_size = (224,224), dimensions = [256,128], dropout = None):
            self.image_shape = image_size + (3,)
            self.augment = Sequential([
                            RandomFlip('horizontal'),
                            RandomRotation(0.2)])
            self.preprocess = PreprocessVGG16(image_size)
            self.base_model = VGG16(include_top=False,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=self.image_shape,
                                    pooling=None,
                                    classes=1000,
                                    classifier_activation='softmax')
            self.global_middle_layer = GlobalAveragePooling2D()
            if dropout is None:
                self.dropout = False
            else:
                self.dropout = True
                self.dropout_layer = Dropout(dropout)
            self.prediction_layers = Sequential([Dense(dim) for dim in dimensions])

        def call(self, x, training=None, mask=None):
            x = self.augment(x)
            x = self.preprocess(x)
            x = self.base_model(x)
            x = self.global_middle_layer(x)
            x = self.dropout_layer(x)
            x = self.prediction_layers(x)
            return x
