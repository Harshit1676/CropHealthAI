"""
Advisory engine: structured management/treatment guidance for every
disease/pest class in the dataset (22 classes across Cashew, Cassava,
Maize and Tomato).

Guidance is intentionally general and safety-conscious: it never gives
specific pesticide dosages, and always points the user to product labels
and local agricultural extension services for chemical control decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from ml.model_utils import parse_class_name

logger = config.get_logger(__name__)

_LABEL_NOTE = "Follow the pesticide label and local agricultural extension recommendations."

# --------------------------------------------------------------------------
# Advisory records keyed by "Crop|Condition" (lower-case, normalized)
# --------------------------------------------------------------------------
ADVISORY_DB: dict[str, dict] = {
    # ---------------------------------------------------------------- Cashew
    "cashew|anthracnose": {
        "symptoms": [
            "Dark, sunken lesions on leaves, shoots, flowers and young nuts",
            "Blighted young shoots and dieback of twigs",
            "Premature fruit/nut drop",
        ],
        "likely_cause": "Fungal infection (Colletotrichum spp.), favored by warm, humid, wet weather.",
        "prevention": [
            "Prune and destroy infected twigs and plant debris",
            "Ensure good canopy spacing/airflow to reduce humidity buildup",
            "Avoid overhead irrigation during flowering/fruiting",
        ],
        "ipm_actions": [
            "Remove and destroy infected plant material promptly",
            "Use disease-free planting material",
            "Encourage natural predators; avoid unnecessary broad-spectrum spraying",
        ],
        "immediate_actions": [
            "Prune out visibly infected shoots and destroy them away from the field",
            "Improve drainage and airflow around affected trees",
        ],
        "general_treatment": f"Copper-based or other recommended fungicides may help manage outbreaks. {_LABEL_NOTE}",
        "monitoring": ["Inspect new flush growth weekly during rainy season", "Track flowering/fruiting stages closely"],
    },
    "cashew|gumosis": {
        "symptoms": ["Gum/resin oozing from bark cracks on trunk and branches", "Bark discoloration and dieback"],
        "likely_cause": "Fungal or bacterial pathogens entering through wounds, often worsened by waterlogging or stem injury.",
        "prevention": [
            "Avoid mechanical injury to bark during field operations",
            "Improve field drainage to prevent waterlogging at the root zone",
            "Whitewash or protect trunk from sun-scald and injury",
        ],
        "ipm_actions": [
            "Clean and disinfect wounds; remove affected bark carefully",
            "Avoid excessive nitrogen fertilization which can increase susceptibility",
        ],
        "immediate_actions": ["Scrape away oozing gum and clean the wound area", "Apply a recommended wound-sealant/fungicidal paste"],
        "general_treatment": f"Wound dressing with a fungicidal paste is commonly recommended. {_LABEL_NOTE}",
        "monitoring": ["Check trunk and major branches monthly for new gum exudation"],
    },
    "cashew|healthy": {
        "symptoms": ["No visible disease or pest symptoms", "Uniform green foliage, normal growth"],
        "likely_cause": "N/A — plant appears healthy.",
        "prevention": ["Maintain regular monitoring", "Follow balanced fertilization and irrigation practices"],
        "ipm_actions": ["Continue routine preventive field sanitation"],
        "immediate_actions": ["No action needed"],
        "general_treatment": "No treatment required.",
        "monitoring": ["Continue weekly visual inspection as part of routine crop management"],
    },
    "cashew|leaf miner": {
        "symptoms": ["Serpentine tunnels/mines visible on leaf surface", "Leaf curling and premature leaf drop in heavy infestation"],
        "likely_cause": "Larvae of leaf-mining moths/flies feeding between leaf surfaces.",
        "prevention": ["Remove and destroy heavily mined leaves", "Encourage natural parasitoid wasps that control leaf miners"],
        "ipm_actions": [
            "Use yellow sticky traps to monitor adult populations",
            "Conserve natural enemies; avoid unnecessary insecticide use",
            "Prune and destroy heavily infested shoots",
        ],
        "immediate_actions": ["Remove and destroy visibly mined leaves", "Monitor surrounding trees for spread"],
        "general_treatment": f"Targeted insecticide application may be warranted in severe infestations. {_LABEL_NOTE}",
        "monitoring": ["Inspect new leaf flush every 1-2 weeks during active growth periods"],
    },
    "cashew|red rust": {
        "symptoms": ["Rust-colored/orange algal patches on leaves and young stems", "Reduced photosynthetic leaf area in heavy infestation"],
        "likely_cause": "Parasitic alga (Cephaleuros spp.), favored by high humidity and poor airflow.",
        "prevention": ["Improve canopy airflow through pruning", "Avoid excess overhead irrigation", "Ensure adequate plant nutrition"],
        "ipm_actions": ["Prune affected branches to improve light/airflow", "Remove heavily infected leaves"],
        "immediate_actions": ["Prune out heavily rust-affected foliage", "Improve drainage and spacing"],
        "general_treatment": f"Copper-based fungicidal sprays are commonly used for algal leaf spots. {_LABEL_NOTE}",
        "monitoring": ["Check canopy during and after humid/rainy periods"],
    },

    # --------------------------------------------------------------- Cassava
    "cassava|bacterial blight": {
        "symptoms": ["Angular water-soaked leaf spots", "Wilting of leaves and shoots", "Gum exudation on stems", "Blighting and dieback"],
        "likely_cause": "Bacterial infection (Xanthomonas spp.), spreads via rain-splash, tools and infected cuttings.",
        "prevention": ["Use certified disease-free planting cuttings", "Practice crop rotation", "Disinfect tools between plants"],
        "ipm_actions": ["Remove and destroy infected plant debris", "Avoid working in fields when foliage is wet", "Rotate with non-host crops"],
        "immediate_actions": ["Remove and destroy severely blighted plants/branches", "Avoid propagating cuttings from infected plants"],
        "general_treatment": f"No reliable chemical cure; focus on cultural and sanitation practices. {_LABEL_NOTE}",
        "monitoring": ["Inspect fields weekly, especially after rain", "Check new plantings from any external cutting source"],
    },
    "cassava|brown spot": {
        "symptoms": ["Circular brown necrotic spots with defined margins on leaves", "Premature leaf yellowing and drop in severe cases"],
        "likely_cause": "Fungal leaf spot pathogen, favored by humid conditions and dense canopy.",
        "prevention": ["Avoid excessive plant density", "Remove fallen infected leaves", "Practice field sanitation between seasons"],
        "ipm_actions": ["Improve field airflow via appropriate spacing", "Remove and destroy heavily spotted leaves"],
        "immediate_actions": ["Remove severely affected lower leaves", "Improve drainage around plants"],
        "general_treatment": f"Fungicidal treatment is rarely necessary; cultural control is usually sufficient. {_LABEL_NOTE}",
        "monitoring": ["Monitor lower canopy leaves during rainy season"],
    },
    "cassava|green mite": {
        "symptoms": ["Yellow/pale mottling on leaves", "Leaf curling, distortion and stunted shoot tips", "Fine webbing in severe infestations"],
        "likely_cause": "Cassava green mite (Mononychellus spp.) infestation, worsened by dry, dusty conditions.",
        "prevention": ["Use tolerant/resistant varieties where available", "Conserve natural predatory mites and insects"],
        "ipm_actions": ["Encourage biological control agents (predatory mites)", "Avoid drought stress through adequate irrigation", "Avoid broad-spectrum insecticides that kill natural enemies"],
        "immediate_actions": ["Prune and destroy heavily infested shoot tips", "Improve irrigation to reduce plant stress"],
        "general_treatment": f"Miticide application may be warranted in severe outbreaks. {_LABEL_NOTE}",
        "monitoring": ["Inspect young shoot tips every 1-2 weeks, especially in dry periods"],
    },
    "cassava|healthy": {
        "symptoms": ["No visible disease or pest symptoms", "Normal leaf color and growth pattern"],
        "likely_cause": "N/A — plant appears healthy.",
        "prevention": ["Maintain routine monitoring", "Use certified planting material"],
        "ipm_actions": ["Continue preventive field sanitation"],
        "immediate_actions": ["No action needed"],
        "general_treatment": "No treatment required.",
        "monitoring": ["Continue weekly visual inspection"],
    },
    "cassava|mosaic": {
        "symptoms": ["Mosaic mottling (light/dark green patches) on leaves", "Leaf distortion and reduced leaf size", "Stunted plant growth"],
        "likely_cause": "Cassava mosaic virus, transmitted by whiteflies and infected cuttings.",
        "prevention": ["Use certified virus-free planting material", "Control whitefly vector populations", "Rogue and destroy infected plants early"],
        "ipm_actions": ["Remove and destroy infected plants promptly (roguing)", "Manage whitefly populations using recommended methods", "Plant resistant/tolerant varieties where available"],
        "immediate_actions": ["Rogue out (remove) infected plants to reduce spread", "Avoid taking cuttings from symptomatic plants"],
        "general_treatment": f"No chemical cure for viral infection; focus on vector control and resistant varieties. {_LABEL_NOTE}",
        "monitoring": ["Scout for whitefly populations regularly", "Inspect new growth for mosaic symptoms every 1-2 weeks"],
    },

    # ----------------------------------------------------------------- Maize
    "maize|fall armyworm": {
        "symptoms": ["Ragged/window-pane feeding holes on leaves", "Sawdust-like frass in the whorl", "Damage concentrated in the funnel/whorl of young plants"],
        "likely_cause": "Larvae of the fall armyworm (Spodoptera frugiperda) feeding on leaves and whorls.",
        "prevention": ["Early planting to avoid peak pest pressure", "Field scouting from crop emergence", "Intercropping/push-pull strategies where feasible"],
        "ipm_actions": ["Handpick and destroy egg masses/larvae in small plots", "Conserve natural enemies (parasitic wasps, predatory beetles)", "Apply biological control agents (e.g. Bt-based products) where available"],
        "immediate_actions": ["Scout whorls for larvae and frass", "Apply targeted control if damage exceeds threshold levels"],
        "general_treatment": f"Insecticide application (including biological options) may be needed for heavy infestation. {_LABEL_NOTE}",
        "monitoring": ["Scout fields twice weekly from seedling to tasseling stage"],
    },
    "maize|grasshoper": {
        "symptoms": ["Irregular chewed margins on leaves", "Defoliation in patches, especially field edges", "Visible grasshoppers/nymphs on plants"],
        "likely_cause": "Grasshopper feeding, often migrating from adjacent vegetation or fallow land.",
        "prevention": ["Manage field margins/weeds that harbor grasshoppers", "Early detection through field-edge scouting"],
        "ipm_actions": ["Use barrier trenches or bait stations where populations are high", "Encourage natural predators (birds, predatory insects)"],
        "immediate_actions": ["Scout field edges where grasshoppers typically enter first", "Target hotspots rather than blanket treatment"],
        "general_treatment": f"Localized insecticide treatment may be needed for large swarms. {_LABEL_NOTE}",
        "monitoring": ["Check field margins weekly, particularly after neighboring vegetation dries"],
    },
    "maize|healthy": {
        "symptoms": ["No visible disease or pest symptoms", "Uniform green healthy foliage"],
        "likely_cause": "N/A — plant appears healthy.",
        "prevention": ["Maintain routine monitoring and balanced fertilization"],
        "ipm_actions": ["Continue preventive field sanitation"],
        "immediate_actions": ["No action needed"],
        "general_treatment": "No treatment required.",
        "monitoring": ["Continue weekly visual inspection"],
    },
    "maize|leaf beetle": {
        "symptoms": ["Small round/irregular holes chewed in leaves", "Skeletonized leaf areas in heavy infestation"],
        "likely_cause": "Adult and larval leaf beetles feeding on foliage.",
        "prevention": ["Field sanitation to remove crop residue that harbors beetles", "Crop rotation with non-host crops"],
        "ipm_actions": ["Handpick beetles in small-scale plots", "Conserve natural predators", "Use trap crops where feasible"],
        "immediate_actions": ["Scout and remove beetles from young, vulnerable plants"],
        "general_treatment": f"Insecticide treatment may be considered for severe, spreading infestations. {_LABEL_NOTE}",
        "monitoring": ["Inspect young plants weekly during early growth stages"],
    },
    "maize|leaf blight": {
        "symptoms": ["Long, elliptical grey-green to tan lesions on leaves", "Lesions may coalesce, causing large areas of blighted tissue"],
        "likely_cause": "Fungal pathogen (e.g. Exserohilum/Helminthosporium spp.), favored by warm humid weather and dense planting.",
        "prevention": ["Use resistant/tolerant maize varieties", "Rotate crops and manage residue", "Avoid excessive plant density"],
        "ipm_actions": ["Remove and destroy heavily blighted residue after harvest", "Ensure balanced fertilization (avoid excess nitrogen)"],
        "immediate_actions": ["Remove severely blighted lower leaves where practical", "Improve field airflow/spacing for future plantings"],
        "general_treatment": f"Fungicide application can help in severe outbreaks, especially on susceptible varieties. {_LABEL_NOTE}",
        "monitoring": ["Scout fields weekly from mid-vegetative stage onward, especially in humid seasons"],
    },
    "maize|leaf spot": {
        "symptoms": ["Small circular to oval necrotic spots on leaves", "Spots may have a yellow halo", "Reduced photosynthetic area in severe cases"],
        "likely_cause": "Fungal leaf spot pathogens, favored by humid, wet conditions and leaf wetness.",
        "prevention": ["Use resistant varieties where available", "Practice crop rotation", "Avoid overhead irrigation late in the day"],
        "ipm_actions": ["Remove and destroy infected crop residue post-harvest", "Ensure balanced soil fertility"],
        "immediate_actions": ["Remove heavily spotted leaves where feasible", "Reduce leaf wetness duration via irrigation timing"],
        "general_treatment": f"Fungicide treatment may be warranted for severe, spreading spots. {_LABEL_NOTE}",
        "monitoring": ["Inspect mid and lower canopy leaves weekly during humid periods"],
    },
    "maize|streak virus": {
        "symptoms": ["Fine, broken chlorotic streaks parallel to leaf veins", "Stunted growth and reduced cob formation in severe cases"],
        "likely_cause": "Maize streak virus, transmitted by leafhoppers.",
        "prevention": ["Use resistant/tolerant maize varieties", "Early planting to avoid peak leafhopper populations", "Control grassy weeds that host the vector"],
        "ipm_actions": ["Manage leafhopper vector populations using recommended methods", "Rogue severely infected plants early in the season"],
        "immediate_actions": ["Remove and destroy severely stunted/streaked plants", "Scout for leafhopper presence"],
        "general_treatment": f"No chemical cure for viral infection; vector control and resistant varieties are key. {_LABEL_NOTE}",
        "monitoring": ["Scout for leafhoppers and streak symptoms weekly from emergence to early vegetative stage"],
    },

    # ---------------------------------------------------------------- Tomato
    "tomato|healthy": {
        "symptoms": ["No visible disease or pest symptoms", "Vigorous green foliage and normal fruit development"],
        "likely_cause": "N/A — plant appears healthy.",
        "prevention": ["Maintain routine monitoring", "Balanced irrigation and fertilization"],
        "ipm_actions": ["Continue preventive field/greenhouse sanitation"],
        "immediate_actions": ["No action needed"],
        "general_treatment": "No treatment required.",
        "monitoring": ["Continue weekly visual inspection"],
    },
    "tomato|leaf blight": {
        "symptoms": ["Dark brown/black lesions, often with concentric rings (target-spot pattern)", "Lesions on leaves, stems and sometimes fruit", "Rapid leaf yellowing and death in severe cases"],
        "likely_cause": "Fungal pathogen (e.g. Alternaria spp. — early blight, or Phytophthora — late blight), favored by humid, wet weather.",
        "prevention": ["Use resistant varieties where available", "Practice crop rotation (avoid solanaceous crops in same plot)", "Stake/prune plants for airflow and avoid overhead irrigation"],
        "ipm_actions": ["Remove and destroy infected lower leaves early", "Mulch to reduce soil splash onto leaves", "Space plants adequately for airflow"],
        "immediate_actions": ["Remove and destroy visibly blighted leaves/stems immediately", "Avoid working in wet foliage to prevent spread"],
        "general_treatment": f"Fungicide application (protectant or systemic) is often necessary once blight is established. {_LABEL_NOTE}",
        "monitoring": ["Scout plants 2-3 times weekly during humid/rainy periods, checking lower leaves first"],
    },
    "tomato|leaf curl": {
        "symptoms": ["Upward or downward curling of leaves", "Leaf yellowing, thickening and stunted growth", "Reduced flowering and fruit set"],
        "likely_cause": "Tomato leaf curl virus, transmitted primarily by whiteflies.",
        "prevention": ["Use resistant/tolerant tomato varieties", "Use whitefly-proof nursery screens", "Remove and destroy infected plants early (roguing)"],
        "ipm_actions": ["Control whitefly populations using recommended integrated methods (yellow sticky traps, reflective mulches)", "Rogue and destroy infected plants promptly", "Avoid planting near heavily infected fields"],
        "immediate_actions": ["Remove and destroy severely curled/stunted plants to reduce virus source", "Inspect for whiteflies on leaf undersides"],
        "general_treatment": f"No chemical cure for viral infection; focus on vector (whitefly) management. {_LABEL_NOTE}",
        "monitoring": ["Monitor whitefly populations with yellow sticky traps weekly"],
    },
    "tomato|septoria leaf spot": {
        "symptoms": ["Small circular spots with dark margins and grey/tan centers", "Tiny black specks (pycnidia) visible within spots", "Lower leaves affected first, progressing upward"],
        "likely_cause": "Fungal pathogen (Septoria lycopersici), favored by wet, humid conditions and leaf wetness.",
        "prevention": ["Rotate crops away from tomato/solanaceous family", "Avoid overhead irrigation; water at the base", "Remove plant debris after harvest"],
        "ipm_actions": ["Remove and destroy infected lower leaves early", "Improve air circulation through staking/pruning", "Mulch to reduce soil splash"],
        "immediate_actions": ["Remove and destroy affected lower leaves promptly", "Avoid overhead watering during outbreaks"],
        "general_treatment": f"Fungicide application may be needed to prevent rapid spread. {_LABEL_NOTE}",
        "monitoring": ["Inspect lower canopy leaves weekly, especially after rain"],
    },
    "tomato|verticulium wilt": {
        "symptoms": ["Yellowing and wilting of lower leaves, often one-sided", "V-shaped chlorotic lesions at leaf margins", "Vascular browning visible in cut stems", "Stunted growth and reduced yield"],
        "likely_cause": "Soil-borne fungal pathogen (Verticillium spp.) infecting the vascular system, persists in soil for years.",
        "prevention": ["Use resistant/tolerant varieties (look for 'V' in disease resistance codes)", "Practice long crop rotation (4+ years) away from susceptible crops", "Solarize soil in heavily infested plots where feasible"],
        "ipm_actions": ["Remove and destroy infected plants including roots", "Avoid moving soil/equipment from infected to clean fields", "Improve soil drainage and avoid plant stress"],
        "immediate_actions": ["Remove and destroy severely wilted plants including root systems", "Avoid replanting susceptible crops in the same soil"],
        "general_treatment": f"No effective curative chemical treatment for established wilt; prevention and resistant varieties are key. {_LABEL_NOTE}",
        "monitoring": ["Inspect for one-sided wilting weekly, especially in previously affected plots"],
    },
}


def _normalize_key(crop: str, condition: str) -> str:
    return f"{crop.strip().lower()}|{condition.strip().lower()}"


def get_advisory(crop: str, condition: str) -> Optional[dict]:
    """Look up advisory info by crop + condition (case-insensitive)."""
    key = _normalize_key(crop, condition)
    record = ADVISORY_DB.get(key)
    if record is None:
        # Try loose matching (e.g. "Leaf Blight" vs "leaf_blight")
        norm_condition = condition.strip().lower().replace("_", " ")
        key2 = _normalize_key(crop, norm_condition)
        record = ADVISORY_DB.get(key2)
    return record


def get_advisory_by_raw_class(raw_class_name: str) -> Optional[dict]:
    meta = parse_class_name(raw_class_name)
    record = get_advisory(meta["crop"], meta["condition"])
    if record:
        return {**meta, **record}
    return None


def list_all_conditions() -> list[dict]:
    """Return a list of {crop, condition} for every advisory entry — used to
    populate the Treatment Advisor page's search/select dropdown."""
    results = []
    for key in ADVISORY_DB:
        crop, condition = key.split("|", 1)
        results.append({"crop": crop.capitalize(), "condition": condition.title()})
    return sorted(results, key=lambda r: (r["crop"], r["condition"]))
