"""
BioCycle - Deep Learning Example
================================

Exemple complet d'entraînement d'un modèle de classification de phase du cycle
en utilisant les données générées par le seeder ML.

Prérequis:
    pip install tensorflow pandas numpy scikit-learn sqlalchemy mysqlclient
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Configuration ==========
DB_HOST = '127.0.0.1'
DB_USER = 'alphadev'
DB_PASSWORD = 'alphadev'
DB_NAME = 'biocycle'
DB_PORT = 3306

# ========== 1. Connexion à la Base de Données ==========
print("🔌 Connexion à la base de données...")
connection_string = f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# ========== 2. Charger les Données ==========
print("📥 Chargement des données ML...")
query = """
SELECT 
    day_of_cycle,
    basal_temp,
    cervical_mucus_score,
    sleep_hours,
    stress_level,
    exercise_minutes,
    water_intake_ml,
    estrogen_pmol_l,
    progesterone_pmol_l,
    fsh_iu_l,
    lh_iu_l,
    has_cramps,
    has_headache,
    has_bloating,
    has_fatigue,
    has_mood_changes,
    has_acne,
    has_libido_change,
    cycle_phase,
    actual_phase
FROM ml_training_data
WHERE is_validated = true AND actual_phase IS NOT NULL
"""

df = pd.read_sql(query, engine)
print(f"✅ Données chargées: {df.shape[0]} samples, {df.shape[1]} features")
print(f"\nDistribution des classes:")
print(df['actual_phase'].value_counts())

# ========== 3. Préparation des Données ==========
print("\n🔧 Préparation des données...")

# Features numériques
numeric_features = [
    'day_of_cycle', 'basal_temp', 'cervical_mucus_score', 
    'sleep_hours', 'stress_level', 'exercise_minutes',
    'water_intake_ml', 'estrogen_pmol_l', 'progesterone_pmol_l',
    'fsh_iu_l', 'lh_iu_l',
    'has_cramps', 'has_headache', 'has_bloating', 'has_fatigue',
    'has_mood_changes', 'has_acne', 'has_libido_change'
]

# Cible
y = df['actual_phase'].values

# Encodage de la cible
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=4)

print(f"Classes encodées: {le.classes_}")

# Features
X = df[numeric_features].values

# Gestion des valeurs manquantes
X = np.nan_to_num(X, nan=0.0)

# Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Features: {X_scaled.shape}")
print(f"✅ Target: {y_categorical.shape}")

# ========== 4. Train/Test Split ==========
print("\n🔀 Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train: {X_train.shape[0]} samples")
print(f"Test: {X_test.shape[0]} samples")

# ========== 5. Construction du Modèle ==========
print("\n🧠 Construction du modèle...")

model = keras.Sequential([
    # Input Layer
    keras.layers.Input(shape=(X_train.shape[1],)),
    
    # Hidden Layers
    keras.layers.Dense(128, activation='relu', name='dense_1'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(64, activation='relu', name='dense_2'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(32, activation='relu', name='dense_3'),
    keras.layers.Dropout(0.1),
    
    keras.layers.Dense(16, activation='relu', name='dense_4'),
    
    # Output Layer
    keras.layers.Dense(4, activation='softmax', name='output')
])

# Compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print(model.summary())

# ========== 6. Entraînement ==========
print("\n🚀 Entraînement du modèle...")

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("✅ Entraînement terminé!")

# ========== 7. Évaluation ==========
print("\n📊 Évaluation du modèle...")

# Évaluation sur l'ensemble de test
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

print(f"\nRésultats sur le test set:")
print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  Loss:      {test_loss:.4f}")

# Prédictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Rapport de classification
print("\nRapport de Classification:")
print(classification_report(
    y_test_classes, y_pred_classes,
    target_names=le.classes_
))

# Matrice de confusion
print("\nMatrice de Confusion:")
cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)

# ========== 8. Visualisation ==========
print("\n📈 Génération des graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train')
axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
axes[0, 0].set_title('Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid()

# Loss
axes[0, 1].plot(history.history['loss'], label='Train')
axes[0, 1].plot(history.history['val_loss'], label='Validation')
axes[0, 1].set_title('Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid()

# Matrice de confusion
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1, 0].set_title('Confusion Matrix')

# Scores
scores = [test_accuracy, test_precision, test_recall, 1-test_loss/test_accuracy]
axes[1, 1].bar(['Accuracy', 'Precision', 'Recall', 'F1-Score'], scores)
axes[1, 1].set_title('Model Metrics')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis='y')

for i in range(4):
    height = scores[i]
    axes[1, 1].text(i, height + 0.01, f'{height:.3f}', ha='center')

plt.tight_layout()
plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
print("✅ Graphiques sauvegardés: model_results.png")

# ========== 9. Prédictions Personnalisées ==========
print("\n🔮 Exemple de Prédiction Personnalisée...")

# Créer un exemple d'utilisateur
sample_data = {
    'day_of_cycle': [14],  # J14 - Ovulation
    'basal_temp': [37.2],
    'cervical_mucus_score': [5],
    'sleep_hours': [8],
    'stress_level': [3],
    'exercise_minutes': [45],
    'water_intake_ml': [2800],
    'estrogen_pmol_l': [350],
    'progesterone_pmol_l': [30],
    'fsh_iu_l': [40],
    'lh_iu_l': [70],
    'has_cramps': [0],
    'has_headache': [0],
    'has_bloating': [0],
    'has_fatigue': [0],
    'has_mood_changes': [0],
    'has_acne': [0],
    'has_libido_change': [1],
}

df_sample = pd.DataFrame(sample_data)
X_sample = scaler.transform(df_sample[numeric_features])
prediction = model.predict(X_sample, verbose=0)

print("\nPrédiction:")
for i, phase in enumerate(le.classes_):
    confidence = prediction[0][i] * 100
    print(f"  {phase}: {confidence:.1f}%")

predicted_phase = le.classes_[np.argmax(prediction[0])]
print(f"\n✅ Phase prédite: {predicted_phase}")

# ========== 10. Sauvegarde du Modèle ==========
print("\n💾 Sauvegarde du modèle...")
model.save('cycle_prediction_model.h5')
print("✅ Modèle sauvegardé: cycle_prediction_model.h5")

# Sauvegarder aussi le scaler et l'encodeur
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Scaler et Label Encoder sauvegardés")

print("\n" + "="*50)
print("✨ Entraînement Complété!")
print("="*50)
print("\nFichiers générés:")
print("  - cycle_prediction_model.h5")
print("  - scaler.pkl")
print("  - label_encoder.pkl")
print("  - model_results.png")

