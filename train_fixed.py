"""
BioCycle - Version Corrigée (Sans TensorFlow)
===========================================
Cette version utilise un Perceptron Multicouche (MLP) de Scikit-Learn.
C'est un réseau de neurones identique en structure à votre modèle Keras,
mais compatible avec tous les processeurs CPU.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier # Alternative à Keras
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
    day_of_cycle, basal_temp, cervical_mucus_score, sleep_hours, stress_level, 
    exercise_minutes, water_intake_ml, estrogen_pmol_l, progesterone_pmol_l, 
    fsh_iu_l, lh_iu_l, has_cramps, has_headache, has_bloating, has_fatigue, 
    has_mood_changes, has_acne, has_libido_change, actual_phase
FROM ml_training_data
WHERE is_validated = true AND actual_phase IS NOT NULL
"""

df = pd.read_sql(query, engine)
print(f"✅ Données chargées: {df.shape[0]} samples")

# ========== 3. Préparation des Données ==========
numeric_features = [
    'day_of_cycle', 'basal_temp', 'cervical_mucus_score', 'sleep_hours', 
    'stress_level', 'exercise_minutes', 'water_intake_ml', 'estrogen_pmol_l', 
    'progesterone_pmol_l', 'fsh_iu_l', 'lh_iu_l', 'has_cramps', 'has_headache', 
    'has_bloating', 'has_fatigue', 'has_mood_changes', 'has_acne', 'has_libido_change'
]

X = df[numeric_features].values
X = np.nan_to_num(X, nan=0.0)

le = LabelEncoder()
y_encoded = le.fit_transform(df['actual_phase'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ========== 5. Construction du Modèle ==========
print("\n🧠 Construction du modèle (Réseau de neurones MLP)...")

# On reproduit la structure Dense(128) -> Dense(64) -> Dense(32) -> Dense(16)
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.2,
    verbose=True,
    random_state=42
)

# ========== 6. Entraînement ==========
print("\n🚀 Entraînement du modèle...")
model.fit(X_train, y_train)
print("✅ Entraînement terminé!")

# ========== 7. Évaluation ==========
print("\n📊 Évaluation du modèle...")
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print(f"\nAccuracy sur le test set: {accuracy:.4f}")
print("\nRapport de Classification:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ========== 8. Visualisation ==========
print("\n📈 Génération des graphiques...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de Confusion')
plt.savefig('model_results.png')
print("✅ Graphique sauvegardé: model_results.png")

# ========== 9. Prédiction Personnalisée ==========
print("\n🔮 Exemple de Prédiction Personnalisée...")
sample_data = np.array([[14, 37.2, 5, 8, 3, 45, 2800, 350, 30, 40, 70, 0, 0, 0, 0, 0, 0, 1]])
X_sample = scaler.transform(sample_data)
prob = model.predict_proba(X_sample)[0]

for i, phase in enumerate(le.classes_):
    print(f"  {phase}: {prob[i]*100:.1f}%")

# ========== 10. Sauvegarde ==========
print("\n💾 Sauvegarde des composants...")
with open('cycle_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✅ Tous les fichiers sont sauvegardés.")