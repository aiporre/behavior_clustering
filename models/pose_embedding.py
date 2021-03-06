from tensorflow.keras.applications import VGG16, vgg16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Conv2D, MaxPooling2D, Lambda
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import tensorflow as tf

class PoseEmbeddings(Model):
        def __init__(self, image_size = (224,224), dimensions = [256,128], dropout = None, use_l2_normalization=True):
            super(PoseEmbeddings, self).__init__()
            self.image_shape = image_size + (3,)
            self.augment = Sequential([
                            RandomFlip('horizontal'),
                            RandomRotation(0.2)])
            # self.base_model = Sequential(name='base_model')
            # self.base_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.image_shape))
            # self.base_model.add(MaxPooling2D((2, 2)))
            # self.base_model.add(Conv2D(64, (3, 3), activation='relu'))
            # self.base_model.add(MaxPooling2D((2, 2)))
            # self.base_model.add(Conv2D(64, (3, 3), activation='relu'))
            self.base_model = VGG16(include_top=False,
                                    weights='imagenet',
                                    input_tensor=None,
                                    input_shape=self.image_shape,
                                    pooling=None,
                                    classifier_activation='softmax')
            self.global_middle_layer = GlobalAveragePooling2D()
            if dropout is None:
                self.dropout = False
            else:
                self.dropout = True
                self.dropout_layer = Dropout(dropout)
            self.prediction_layers = Sequential([Dense(dim) for dim in dimensions])
            self.use_l2_normalization = use_l2_normalization
            if use_l2_normalization:
                self.norm_l2 = Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings



        def call(self, x, training=None, mask=None):
            x = self.augment(x)
            x = vgg16.preprocess_input(x * 255)
            x = tf.image.resize(x, self.image_shape[:2])
            x = self.base_model(x)
            x = self.global_middle_layer(x)
            if self.dropout:
                x = self.dropout_layer(x)
            x = self.prediction_layers(x)
            if self.use_l2_normalization:
                x = self.norm_l2(x)
            return x
