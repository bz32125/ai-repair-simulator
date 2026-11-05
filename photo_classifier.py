"""Load the trained photo classifier, if it exists. The app falls back to text when it does not."""
import json
from pathlib import Path

MODEL = Path(__file__).parent / "models" / "classifier.pt"
CLASSES = MODEL.with_name("classes.json")
MIN_CONFIDENCE = 0.6
_model = None


def available():
    if not (MODEL.exists() and CLASSES.exists()):
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def predict(file):
    """Return (object id, confidence), or (None, 0.0) when unsure or unavailable."""
    global _model
    if file is None or not available():
        return None, 0.0
    import torch
    from PIL import Image
    from train_classifier import EVAL_TF, build_model
    classes = json.loads(CLASSES.read_text())
    if _model is None:
        _model = build_model(len(classes))
        _model.load_state_dict(torch.load(MODEL, map_location="cpu"))
        _model.eval()
    x = EVAL_TF(Image.open(file).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(_model(x), 1)[0]
    conf, i = p.max(0)
    return (classes[i], conf.item()) if conf.item() >= MIN_CONFIDENCE else (None, conf.item())
