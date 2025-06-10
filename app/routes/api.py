from flask import Blueprint, request, jsonify
from backend.app.utils.data_processor import load_patient_data
from backend.app.utils.image_processor import load_image_from_row, preprocess_image
import torch
from backend.app.models.qvt_model import QVTModel
import os

bp = Blueprint('api', __name__)

# Load the trained model at startup
model = QVTModel()
model_path = os.path.join(os.path.dirname(__file__), '../models/qvt_trained.pth')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

@bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Backend is running!"})

@bp.route('/predict', methods=['POST'])
def predict():
    id_value = request.json.get('Unnamed: 0')
    which_eye = request.json.get('eye', 'od')
    use_path = request.json.get('use_path', True)

    df = load_patient_data(which_eye)
    row = df[df['Unnamed: 0'] == id_value]
    if row.empty:
        return jsonify({'error': 'ID not found'}), 404
    row = row.iloc[0]

    try:
        img = load_image_from_row(row, use_path=use_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Preprocess and predict
    img_tensor = preprocess_image(img)
    with torch.no_grad():
        output = model(img_tensor)
        pred = int((output > 0.5).item())
        label = "glaucoma" if pred == 1 else "no glaucoma"
        confidence = float(output.item())

    return jsonify({
        "result": "success",
        "prediction": label,
        "confidence": confidence
    })