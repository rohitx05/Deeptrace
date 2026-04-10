import torch
from models.generator_head import GeneratorFingerprintHead
from models.detector import DeepfakeDetector
from explainability.attention_viz import AttentionVisualizer

def test_generator():
    print("Testing GeneratorFingerprintHead...")
    head = GeneratorFingerprintHead()
    x = torch.randn(2, 1280)
    out = head(x)
    print("GeneratorFingerprintHead forward pass OK")

def test_attention():
    print("Testing AttentionVisualizer...")
    model = DeepfakeDetector()
    # Mock a forward pass to populate attn_weights
    x_img = torch.randn(2, 3, 160, 160)
    x_dct = torch.randn(2, 3, 160, 160)
    x_frames = torch.randn(2, 8, 3, 160, 160)
    x_dct_frames = torch.randn(2, 8, 3, 160, 160)
    
    print("Running detector video mode...")
    out = model(frames=x_frames, dct_frames=x_dct_frames, mode="video")
    
    vis = AttentionVisualizer(model)
    res = vis.generate_full_report(save_dir="results_test/")
    print(f"AttentionVisualizer OK! Generated {res}")

if __name__ == "__main__":
    try:
        test_generator()
        test_attention()
        print("ALL TESTS PASSED WITH NO ERRORS")
    except Exception as e:
        import traceback
        traceback.print_exc()
