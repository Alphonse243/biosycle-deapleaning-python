import pickle
import pandas as pd

# 1. Charger les fichiers sauvegardés
with open('cycle_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# 2. Créer une donnée d'exemple (J14, mucus fertile, temp élevée...)
nouvelle_donnee = pd.DataFrame([{
    'day_of_cycle': 14, 'basal_temp': 37.1, 'cervical_mucus_score': 4,
    'sleep_hours': 8, 'stress_level': 2, 'exercise_minutes': 30,
    'water_intake_ml': 2000, 'estrogen_pmol_l': 300, 'progesterone_pmol_l': 10,
    'fsh_iu_l': 15, 'lh_iu_l': 40, 'has_cramps': 0, 'has_headache': 0,
    'has_bloating': 0, 'has_fatigue': 0, 'has_mood_changes': 0,
    'has_acne': 0, 'has_libido_change': 1
}])

# 3. Prédire
X_scaled = scaler.transform(nouvelle_donnee)
prediction_code = model.predict(X_scaled)
phase_nom = le.inverse_transform(prediction_code)

print(f"La phase prédite est : {phase_nom[0]}")