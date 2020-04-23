from keras.applications import VGG16, ResNet50, InceptionV3, Xception, DenseNet121
from keras.utils import plot_model

# https://github.com/keras-team/keras-applications/tree/master/keras_applications

vgg16 = VGG16(weights=None)
vgg16.summary()
plot_model(vgg16, "vgg16.png", show_layer_names= True, show_shapes= True)

resnet50 = ResNet50(weights=None)
resnet50.summary()
plot_model(resnet50, "resnet50.png", show_layer_names= True, show_shapes= True)

inception = InceptionV3(weights=None)
inception.summary()
plot_model(inception, "inception.png", show_layer_names= True, show_shapes= True)

xception = Xception(weights=None)
xception.summary()
plot_model(xception, "xception.png", show_layer_names= True, show_shapes= True)


densenet = DenseNet121(weights=None)
densenet.summary()
plot_model(densenet, "densenet.png", show_layer_names= True, show_shapes= True)
