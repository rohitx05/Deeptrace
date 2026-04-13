"""Dry-run test: verify model output shapes and probability values."""
import sys, torch, yaml, math
sys.path.insert(0, '.')

from calibration import ModelWithTemperature, load_calibration_dict
from models.detector import DeepfakeDetector
from utils.checkpoint import load_checkpoint

device = torch.device('cuda')

with open('configs/model_config.yaml') as f:
    model_config = yaml.safe_load(f)

model = DeepfakeDetector(config=model_config)
load_checkpoint('checkpoints/kaggle_realfake/best_model.pth', model, device=device)
model.to(device).eval()

wrapped = ModelWithTemperature(model)
calib = load_calibration_dict('checkpoints/kaggle_realfake/calibration.json')
wrapped.set_temperature_value(calib['temperature'])
wrapped.to(device).eval()

threshold = float(calib.get('threshold') or calib.get('optimal_threshold', 0.5))
print(f'Temperature : {wrapped.temperature_value:.4f}')
print(f'Threshold   : {threshold}')

# --- Dry-run with random input (float32, no AMP) ---
dummy_img = torch.randn(1, 3, 224, 224).float().to(device)
dummy_dct = torch.randn(1, 3, 224, 224).float().to(device)

with torch.no_grad():
    preds = wrapped(images=dummy_img, dct=dummy_dct, mode='image')

raw_logit = float(preds['binary_logit'].item())
scaled    = float(preds['scaled_binary_logit'].item())
prob      = float(preds['binary_pred'].item())
manip     = preds['manipulation_pred']

print(f'raw_logit        = {raw_logit:.4f}')
print(f'scaled_logit     = {scaled:.4f}')
print(f'binary_pred(prob)= {prob:.4f}')
print(f'manip shape      = {manip.shape}  value = {manip}')

verdict = 'FAKE' if prob > threshold else 'REAL'
if prob > threshold:
    conf = (prob - threshold) / max(1.0 - threshold, 1e-8)
else:
    conf = (threshold - prob) / max(threshold, 1e-8)

print(f'Verdict     : {verdict}  (prob={prob:.4f} thr={threshold})')
print(f'Confidence  : {conf*100:.1f}%')
print()
print('NaN check -- binary_pred NaN:', math.isnan(prob))
print('NaN check -- scaled NaN:', math.isnan(scaled))
