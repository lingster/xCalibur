"""Publication regressions: private files, notebook output and inaccessible links."""
import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_public", ROOT / "tools" / "check_public.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_private_paths_and_values_are_rejected_without_echoing_values():
    for path in ["garage/run.json", ".aws/config", "docs/.env", "data/credentials.json"]:
        assert audit.forbidden_path(path)
    credential = "ghp_" + "x" * 36
    findings = audit.sensitive_text("a\n" + credential)
    assert findings == [(2, "credential token")]
    assert credential not in str(findings)


def test_notebook_text_scans_outputs_but_not_raster_payloads():
    credential = "hf_" + "x" * 30
    nb = {"cells": [{"source": ["print('result')"], "outputs": [
        {"data": {"image/png": credential, "text/plain": "safe"}},
        {"output_type": "stream", "text": credential},
    ]}]}
    assert len(audit.sensitive_text(audit.notebook_text(nb))) == 1
    nb["cells"][0]["outputs"].pop()
    assert audit.sensitive_text(audit.notebook_text(nb)) == []


def test_existing_private_file_is_not_a_valid_public_link(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# A public guide\n")
    (tmp_path / "garage").mkdir()
    (tmp_path / "garage" / "result.md").write_text("private")
    files = {"README.md", "docs/guide.md"}
    assert audit.check_links(tmp_path, "README.md", "[guide](docs/guide.md#a-public-guide)", files) == []
    assert audit.check_links(tmp_path, "README.md", "[private](garage/result.md)", files)
    assert audit.check_links(tmp_path, "README.md", "[bad](docs/guide.md#absent)", files)


def test_links_and_metadata_are_checked_in_notebooks(tmp_path):
    import json
    (tmp_path / "notebooks").mkdir()
    name = "notebooks/01_demo.ipynb"
    nb = {"metadata": {"author": "private-fixture-person"}, "cells": [
        {"cell_type": "markdown", "source": "[missing](missing.md)"},
        {"cell_type": "code", "source": "1", "execution_count": None, "outputs": []},
    ]}
    (tmp_path / name).write_text(json.dumps(nb))
    findings = audit.inspect(tmp_path, {name}, ["private-fixture-person"])
    assert any("private term" in reason for _, reason in findings)
    assert any("not in public files" in reason for _, reason in findings)
    assert any("unexecuted" in reason for _, reason in findings)


def test_attention_uses_distributed_compiler_helpers(monkeypatch):
    # Isolate the import bridge without loading Torch, Triton or a CUDA driver.
    path = ROOT / "kernels" / "tg4dac_sm120" / "attention.py"
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ws_module")
    calls = []
    driver = SimpleNamespace(load=lambda cubin: "loaded:" + cubin)
    compiler = SimpleNamespace(compile=lambda src, arch: (calls.append((src, arch)) or ("cubin", "log")))
    helper = SimpleNamespace(_setup_torch=lambda: None, Driver=lambda: driver, NVRTC=lambda: compiler)
    monkeypatch.setitem(sys.modules, "mlp", helper)
    ns = {"__package__": None, "_WS": {}, "torch": SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (12, 0)))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "attention_helper", "exec"), ns)
    result = ns["ws_module"]("fixture", "cuda source")
    assert calls == [("cuda source", "sm_120")]
    assert result[2] == "loaded:cubin"
    assert ns["ws_module"]("fixture", "ignored cached input") is result
    assert len(calls) == 1
