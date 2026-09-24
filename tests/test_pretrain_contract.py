import ast
from pathlib import Path


def test_pretrain_model_has_no_physics_or_lsp_heads() -> None:
    tree = ast.parse(Path("src/models/pretrain.py").read_text(encoding="utf-8"))
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert "PretrainCoreLossModel" in names
    assert "PhysicsInformedResidualLayer" not in names
    text = Path("src/models/pretrain.py").read_text(encoding="utf-8")
    assert "rul_head" not in text
    assert "unc_head" not in text

