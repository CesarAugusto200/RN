import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from datetime import datetime

# ------------------------- CONFIGURACIONES -------------------------
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
COLOR_PATH = "images"
GRISES_PATH = "images_grises"
MODEL_PATH = "clasificador_cascaras.h5"
CAPTURE_DIR = "capturas"

# -------------------------------------------------------------------
# 🔄 GENERAR VERSIÓN EN ESCALA DE GRISES AUTOMÁTICAMENTE
def crear_dataset_en_grises(color_path, grises_path):
    print("🔧 Generando imágenes en escala de grises (si no existen)...")
    for class_name in os.listdir(color_path):
        color_class_path = os.path.join(color_path, class_name)
        grises_class_path = os.path.join(grises_path, class_name)

        if not os.path.isdir(color_class_path):
            continue

        os.makedirs(grises_class_path, exist_ok=True)

        for filename in os.listdir(color_class_path):
            color_img_path = os.path.join(color_class_path, filename)
            grises_img_path = os.path.join(grises_class_path, filename)

            if os.path.exists(grises_img_path):
                continue

            img_color = cv2.imread(color_img_path)
            if img_color is None:
                print(f"❌ No se pudo cargar: {color_img_path}")
                continue

            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            img_gray_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            cv2.imwrite(grises_img_path, img_gray_rgb)
            print(f"✅ {grises_img_path}")

crear_dataset_en_grises(COLOR_PATH, GRISES_PATH)

def verificar_similitud_ssim(color_path, grises_path, output_csv="similitud.csv"):
    print("📊 Comparando similitud entre imágenes color y grises...")

    resultados = []

    for class_name in os.listdir(color_path):
        color_class_path = os.path.join(color_path, class_name)
        grises_class_path = os.path.join(grises_path, class_name)

        if not os.path.isdir(color_class_path):
            continue

        for filename in os.listdir(color_class_path):
            color_img_path = os.path.join(color_class_path, filename)
            grises_img_path = os.path.join(grises_class_path, filename)

            if not os.path.exists(grises_img_path):
                continue

            img_color = cv2.imread(color_img_path)
            img_gris = cv2.imread(grises_img_path)

            if img_color is None or img_gris is None:
                continue

            img_color_gray = cv2.cvtColor(cv2.resize(img_color, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)
            img_gris_gray = cv2.cvtColor(cv2.resize(img_gris, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2GRAY)

            score, _ = ssim(img_color_gray, img_gris_gray, full=True)

            resultados.append({
                "clase": class_name,
                "archivo": filename,
                "ssim": round(score, 4)
            })

    df = pd.DataFrame(resultados)
    df.to_csv(output_csv, index=False)
    print(f"✅ Similitud calculada. Resultados guardados en: {output_csv}")
    print(df.groupby("clase")["ssim"].mean().round(4))

# 🔁 Función para combinar dos generadores (color + grises)
def combinar_generadores(gen1, gen2):
    while True:
        x1, y1 = next(gen1)
        x2, y2 = next(gen2)
        x_comb = np.concatenate((x1, x2), axis=0)
        y_comb = np.concatenate((y1, y2), axis=0)
        yield x_comb, y_comb

# PREPROCESAMIENTO Y GENERADORES
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_color = train_datagen.flow_from_directory(
    COLOR_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE // 2,
    class_mode="categorical",
    subset="training"
)

val_color = train_datagen.flow_from_directory(
    COLOR_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE // 2,
    class_mode="categorical",
    subset="validation"
)

train_gris = train_datagen.flow_from_directory(
    GRISES_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE // 2,
    class_mode="categorical",
    subset="training"
)

val_gris = train_datagen.flow_from_directory(
    GRISES_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE // 2,
    class_mode="categorical",
    subset="validation"
)

train_data = combinar_generadores(train_color, train_gris)
val_data = combinar_generadores(val_color, val_gris)

steps_per_epoch = (train_color.samples + train_gris.samples) // BATCH_SIZE
validation_steps = (val_color.samples + val_gris.samples) // BATCH_SIZE

class_labels = list(train_color.class_indices.keys())
print("Clases detectadas:")
for idx, clase in enumerate(class_labels):
    print(f"{idx}: {clase}")

# MODELO CNN (MobileNetV2 + Fine Tuning)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
predictions = layers.Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ENTRENAMIENTO
history = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS
)

# GRÁFICAS
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# GUARDADO
model.save(MODEL_PATH)
print("✅ Modelo guardado como 'clasificador_cascaras.h5'")

# CLASIFICACIÓN EN TIEMPO REAL + GUARDADO DE IMÁGENES
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Modelo cargado.")
else:
    print("No se encontró el modelo entrenado. Entrénalo primero.")
    model = None

def clasificar_y_mostrar(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = preprocess_input(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = class_labels[class_idx]

    # Mostrar en pantalla
    cv2.putText(frame, f"Cáscara: {class_name}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Guardar imagen clasificada
    class_dir = os.path.join(CAPTURE_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    save_path = os.path.join(class_dir, f"{timestamp}.jpg")
    cv2.imwrite(save_path, frame)

    return frame

if model:
    cap = cv2.VideoCapture(0)
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = clasificar_y_mostrar(frame)
        cv2.imshow("Clasificador de Cáscaras - Cámara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
