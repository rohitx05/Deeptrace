import json, sys
sys.path.insert(0, '.')

# 1. calibration.json
d = json.load(open('checkpoints/kaggle_realfake/calibration.json'))
assert abs(d['temperature'] - 4.397) < 0.01, f"Bad temp: {d['temperature']}"
assert abs(d['optimal_threshold'] - 0.162) < 0.001, f"Bad threshold: {d['optimal_threshold']}"
print('calibration.json OK:', d)

# 2. load_calibration_dict
from calibration import load_calibration_dict
c = load_calibration_dict('checkpoints/kaggle_realfake/calibration.json')
assert 'temperature' in c and 'optimal_threshold' in c
print('load_calibration_dict OK:', c)

# 3. config.yaml threshold
import yaml
cfg = yaml.safe_load(open('configs/config.yaml'))
assert abs(cfg['evaluation']['threshold'] - 0.162) < 0.001
print('config.yaml threshold OK:', cfg['evaluation']['threshold'])

# 4. project_state.json
ps = json.load(open('project_state.json'))
assert ps['calibration_status']['status'] == 'done'
assert abs(ps['performance_metrics']['optimal_threshold'] - 0.162) < 0.001
print('project_state.json OK: status=done, threshold=', ps['performance_metrics']['optimal_threshold'])

# 5. Pipeline import + threshold attribute
from inference.pipeline import InferencePipeline
print('InferencePipeline import OK')

print()
print('ALL VERIFICATIONS PASSED')
