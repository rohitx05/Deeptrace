import importlib
modules = [
    ("models.generator_head", "generator_head"),
    ("explainability.attention_viz", "attention_viz"),
    ("explainability.gradcam", "gradcam"),
    ("training.losses", "losses"),
    ("models.clip_alignment", "clip_alignment"),
    ("datasets.celebdf", "celebdf")
]

for mod, name in modules:
    try:
        importlib.import_module(mod)
        print(f"Successfully imported {name}")
    except Exception as e:
        print(f"ERROR importing {name}: {e}")
        import traceback
        traceback.print_exc()

print("DONE")
