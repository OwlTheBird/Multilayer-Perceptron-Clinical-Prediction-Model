"""
FastAPI Backend for Clinical Prediction Models
=============================================
Serves predictions from Classification, Regression, and MTL models.
Each model has its own architecture and feature set.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import os

app = FastAPI(
    title="Clinical Prediction API",
    description="API for kidney and multi-task clinical predictions",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL ARCHITECTURES (must match training)
# ============================================================================

class ClassificationMLP(nn.Module):
    """Kidney Disease Classification Model (3 classes) - 30 features"""
    def __init__(self, input_size=30, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, num_classes)
        self.drop = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.drop(torch.relu(self.bn1(self.fc1(x))))
        x = self.drop(torch.relu(self.bn2(self.fc2(x))))
        x = self.drop(torch.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


class ImprovedRegressionModel(nn.Module):
    """Kidney ACR Regression Model - 35 features"""
    def __init__(self, input_size=35, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x)


class SharedBottomMTL(nn.Module):
    """Multi-Task Learning Model - 30 features, 4 outputs"""
    def __init__(self, num_continuous=30, hidden_dim=256):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(num_continuous)
        self.shared_backbone = nn.Sequential(
            nn.Linear(num_continuous, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.head_cardio = nn.Linear(hidden_dim, 1)
        self.head_metabolic = nn.Linear(hidden_dim, 5)
        self.head_kidney = nn.Linear(hidden_dim, 2)
        self.head_liver = nn.Linear(hidden_dim, 1)

    def forward(self, x_cont):
        x = self.input_bn(x_cont)
        z = self.shared_backbone(x)
        return (self.head_cardio(z), self.head_metabolic(z), 
                self.head_kidney(z), self.head_liver(z))


# ============================================================================
# FEATURE DEFINITIONS FOR EACH MODEL
# ============================================================================

CLASSIFICATION_FEATURES = [
    "age", "income_ratio", "body_mass_index", "height_cm", "heart_rate_bpm",
    "white_blood_cells_count", "platelets_count", "hemoglobin_g_dl",
    "mean_corpuscular_volume_fL", "creatinine_mg_dl", "liver_ast_U_L",
    "bilirubin_mg_dl", "liver_ggt_U_L", "uric_acid_mg_dl", "sodium_mmol_L",
    "potassium_mmol_L", "cholesterol_mg_dl", "alcohol_drinks_per_week",
    "gender_1.0", "gender_2.0", "ethnicity_1.0", "ethnicity_2.0",
    "ethnicity_3.0", "ethnicity_4.0", "ethnicity_6.0", "ethnicity_7.0",
    "smoking_status_1.0", "smoking_status_2.0", "smoking_status_3.0",
    "smoking_status_nan"
]

REGRESSION_FEATURES = [
    "age", "income_ratio", "body_mass_index", "height_cm", "heart_rate_bpm",
    "white_blood_cells_count", "platelets_count", "hemoglobin_g_dl",
    "mean_corpuscular_volume_fL", "creatinine_mg_dl", "liver_ast_U_L",
    "bilirubin_mg_dl", "liver_ggt_U_L", "uric_acid_mg_dl", "sodium_mmol_L",
    "potassium_mmol_L", "cholesterol_mg_dl", "alcohol_drinks_per_week",
    "gender_1.0", "gender_2.0", "ethnicity_1.0", "ethnicity_2.0",
    "ethnicity_3.0", "ethnicity_4.0", "ethnicity_6.0", "ethnicity_7.0",
    "smoking_status_1.0", "smoking_status_2.0", "smoking_status_3.0",
    "smoking_status_nan", "has_cardiovascular_disease", "high_waist_circumference",
    "high_triglycerides_mg_dl", "low_hdl_mg_dl", "high_blood_pressure"
]

MTL_FEATURES = CLASSIFICATION_FEATURES.copy()

CLASS_NAMES = ["Normal", "Microalbuminuria", "Macroalbuminuria"]


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class PredictionRequest(BaseModel):
    features: List[float]

class ClassificationResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: Dict[str, float]

class RegressionResponse(BaseModel):
    prediction: float
    acr_value: float
    risk_category: str

class MTLResponse(BaseModel):
    cardiovascular_disease: Dict[str, Any]
    metabolic_syndrome: Dict[str, Any]
    kidney_dysfunction: Dict[str, Any]
    liver_dysfunction: Dict[str, Any]


# ============================================================================
# GLOBAL MODEL INSTANCES
# ============================================================================

models = {"classification": None, "regression": None, "mtl": None}


def load_models():
    global models
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load Classification Model (30 features)
    try:
        path = os.path.join(base_path, "Classification Model", "best_model.pth")
        if os.path.exists(path):
            models["classification"] = ClassificationMLP(30, 3)
            models["classification"].load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True))
            models["classification"].eval()
            print("✓ Classification model loaded (30 features)")
    except Exception as e:
        print(f"✗ Classification error: {e}")
    
    # Load Regression Model (34 features)
    try:
        path = os.path.join(base_path, "Regression Neural Network", "best_kidney_model.pth")
        if os.path.exists(path):
            models["regression"] = ImprovedRegressionModel(35, 1)
            models["regression"].load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True))
            models["regression"].eval()
            print("✓ Regression model loaded (34 features)")
    except Exception as e:
        print(f"✗ Regression error: {e}")
    
    # Load MTL Model (30 features)
    try:
        path = os.path.join(base_path, "3. Model", "trained_model.pth")
        if os.path.exists(path):
            models["mtl"] = SharedBottomMTL(30, 256)
            models["mtl"].load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True))
            models["mtl"].eval()
            print("✓ MTL model loaded (30 features)")
    except Exception as e:
        print(f"✗ MTL error: {e}")


@app.on_event("startup")
async def startup():
    load_models()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "models_available": {
            "classification": models["classification"] is not None,
            "regression": models["regression"] is not None,
            "mtl": models["mtl"] is not None
        }
    }


@app.get("/features/{model_name}")
async def get_features(model_name: str):
    """Get features required for a specific model"""
    if model_name == "classification":
        return {"features": CLASSIFICATION_FEATURES, "count": len(CLASSIFICATION_FEATURES)}
    elif model_name == "regression":
        return {"features": REGRESSION_FEATURES, "count": len(REGRESSION_FEATURES)}
    elif model_name == "mtl":
        return {"features": MTL_FEATURES, "count": len(MTL_FEATURES)}
    raise HTTPException(status_code=404, detail="Model not found")


@app.post("/predict/classification", response_model=ClassificationResponse)
async def predict_classification(request: PredictionRequest):
    if models["classification"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(request.features)}")
    
    with torch.no_grad():
        x = torch.tensor([request.features], dtype=torch.float32)
        output = models["classification"](x)
        probs = torch.softmax(output, dim=1)[0]
        pred = output.argmax(dim=1).item()
    
    return ClassificationResponse(
        prediction=pred,
        class_name=CLASS_NAMES[pred],
        probabilities={name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}
    )


@app.post("/predict/regression", response_model=RegressionResponse)
async def predict_regression(request: PredictionRequest):
    if models["regression"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.features) != 35:
        raise HTTPException(status_code=400, detail=f"Expected 35 features, got {len(request.features)}")
    
    with torch.no_grad():
        x = torch.tensor([request.features], dtype=torch.float32)
        output = models["regression"](x)
        pred = output.item()
    
    acr_value = np.exp(pred)
    risk = "Normal" if acr_value < 30 else ("Microalbuminuria" if acr_value < 300 else "Macroalbuminuria")
    
    return RegressionResponse(prediction=pred, acr_value=round(acr_value, 2), risk_category=risk)


@app.post("/predict/mtl", response_model=MTLResponse)
async def predict_mtl(request: PredictionRequest):
    if models["mtl"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(request.features)}")
    
    with torch.no_grad():
        x = torch.tensor([request.features], dtype=torch.float32)
        cardio, metabolic, kidney, liver = models["mtl"](x)
        
        cardio_prob = torch.sigmoid(cardio).item()
        metabolic_probs = torch.sigmoid(metabolic)[0].tolist()
        kidney_probs = torch.sigmoid(kidney)[0].tolist()
        liver_prob = torch.sigmoid(liver).item()
    
    return MTLResponse(
        cardiovascular_disease={"probability": round(cardio_prob, 4), "risk": "High" if cardio_prob > 0.5 else "Low"},
        metabolic_syndrome={
            "waist": round(metabolic_probs[0], 4),
            "triglycerides": round(metabolic_probs[1], 4),
            "hdl": round(metabolic_probs[2], 4),
            "blood_pressure": round(metabolic_probs[3], 4),
            "glucose": round(metabolic_probs[4], 4)
        },
        kidney_dysfunction={
            "at_least_micro": round(kidney_probs[0], 4),
            "macro": round(kidney_probs[1], 4),
            "stage": "Macro" if kidney_probs[1] > 0.5 else ("Micro" if kidney_probs[0] > 0.5 else "Normal")
        },
        liver_dysfunction={"probability": round(liver_prob, 4), "risk": "High" if liver_prob > 0.5 else "Low"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
