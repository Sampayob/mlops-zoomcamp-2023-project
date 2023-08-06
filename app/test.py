"""prediction test script"""

import requests

inference = {
    'erythema': 2,
    'scaling': 2,
    'definite_borders': 2,
    'itching': 2,
    'koebner_phenomenon': 2,
    'polygonal_papules': 2,
    'follicular_papules': 0,
    'oral_mucosal_involvement': 2,
    'knee_and_elbow_involvement': 1,
    'scalp_involvement': 0,
    'family_history': 1,
    'melanin_incontinence': 1,
    'eosinophils_infiltrate': 0,
    'PNL_infiltrate': 0,
    'fibrosis_papillary_dermis': 0,
    'exocytosis': 2,
    'acanthosis': 3,
    'hyperkeratosis': 1,
    'parakeratosis': 1,
    'clubbing_rete_ridges': 0,
    'elongation_rete_ridges': 0,
    'thinning_suprapapillary_epidermis': 0,
    'spongiform_pustule': 0,
    'munro_microabcess': 0,
    'focal_hypergranulosis': 2,
    'disappearance_granular_layer': 2,
    'vacuolisation_damage_basal_layer': 2,
    'spongiosis': 3,
    'saw_tooth_appearance_retes': 2,
    'follicular_horn_plug': 0,
    'perifollicular_parakeratosis': 0,
    'inflammatory_mononuclear_infiltrate': 3,
    'band_like_infiltrate': 2,
    'age': '40'
    }

url = 'http://localhost:9696/predict'
response = requests.post(url, json=inference)
print(response.json())