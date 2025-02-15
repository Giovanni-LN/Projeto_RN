import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, concatenate, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.data import Dataset

# Caminho das imagens
train_dir = r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\train"
val_dir = r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\val"
test_dir = r"C:\Users\caiof\Documentos\Projeto_RN\MyFood Dataset\test"

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

# Geradores de imagens
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
val_generator = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

# Carregar as redes pré-treinadas
vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
resnet50 = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
mobilenetv2 = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Congelar camadas das redes pré-treinadas
for layer in vgg16.layers:
    layer.trainable = False

for layer in resnet50.layers:
    layer.trainable = False

for layer in mobilenetv2.layers:
    layer.trainable = False

# Inputs para múltiplas redes
input_vgg16 = Input(shape=(224, 224, 3))
input_resnet50 = Input(shape=(224, 224, 3))
input_mobilenetv2 = Input(shape=(224, 224, 3))

# Extrair características e combinar as saídas das redes
vgg16_output = Flatten()(vgg16(input_vgg16))
resnet50_output = Flatten()(resnet50(input_resnet50))
mobilenetv2_output = Flatten()(mobilenetv2(input_mobilenetv2))

combined_output = concatenate([vgg16_output, resnet50_output, mobilenetv2_output])

# Adicionar camadas personalizadas
x = Dense(512, activation="relu")(combined_output)
x = Dropout(0.5)(x)
x = Dense(10, activation="softmax")(x)  # 10 classes de comida

# Criar o modelo final
model = Model(inputs=[input_vgg16, input_resnet50, input_mobilenetv2], outputs=x)

# Compilar o modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Ajustar o gerador de entrada para múltiplas redes
def multi_input_generator(generator):
    while True:
        x, y = next(generator)
        yield [x, x, x], y

# Treinar o modelo
history = model.fit(
    multi_input_generator(train_generator),
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=multi_input_generator(val_generator),
    validation_steps=len(val_generator)
)

# Plotar a acurácia
plt.plot(history.history["accuracy"], label="Treino")
plt.plot(history.history["val_accuracy"], label="Validação")
plt.legend()
plt.title("Acurácia durante o treinamento")
plt.show()
