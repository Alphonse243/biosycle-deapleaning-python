"""
BioCycle - Version Compatible (Scikit-Learn)
===========================================
Cette version remplace TensorFlow par Scikit-Learn pour éviter l'erreur 
'Instruction non permise' sur les processeurs sans instructions AVX.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier # Remplaçant de TensorFlow
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
try:
    connection_string = f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(connection_string)
except Exception as e:
    print(f"❌ Erreur de connexion : {e}")
    exit()

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
print(f"✅ Données chargées: {df.shape[0]} lignes")

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
y = le.fit_transform(df['actual_phase'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 5. Construction du Modèle (MLP = Réseau de Neurones) ==========
print("\n🧠 Entraînement du réseau de neurones (MLPClassifier)...")

# Cette structure imite votre modèle TensorFlow original
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32, 16), 
    activation='relu', 
    solver='adam', 
    alpha=0.001,       # Régularisation
    batch_size=32, 
    learning_rate_init=0.001,
    max_iter=200,      # Nombre d'époques max
    early_stopping=True,
    validation_fraction=0.2,
    verbose=True,
    random_state=42
)

model.fit(X_train, y_train)

# ========== 6. Évaluation ==========
print("\n📊 Évaluation du modèle...")
y_pred = model.predict(X_test)

print("\nRapport de Classification:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ========== 7. Visualisation & Sauvegarde ==========
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Matrice de Confusion')
plt.ylabel('Réel')
plt.xlabel('Prédit')
plt.savefig('model_results.png')
print("✅ Graphique sauvegardé: model_results.png")

# Sauvegarde des objets
with open('cycle_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("\n✨ Terminé ! Modèle sauvegardé sous 'cycle_model.pkl'")