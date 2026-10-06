import math
import os
import sys

import numpy as np
import pytest
from utils import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "gguf-py"))
import gguf  # noqa: E402

server: ServerProcess


def local_model(server: ServerProcess, env: str) -> ServerProcess:
    """Use a local model file when the environment variable is set (builds without HTTPS)."""
    path = os.environ.get(env)
    if path:
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = path
    return server


def tiny() -> ServerProcess:
    return local_model(ServerPreset.tinyllama2(), "TEST_MODEL_STORIES260K")


def write_head(path: str, n_embd: int, layer: int, weight: np.ndarray, bias: float, question_id: str = "relevant") -> str:
    w = gguf.GGUFWriter(path, "classifier")
    w.add_type("classifier")
    w.add_uint32("classifier.layer", layer)
    w.add_string("classifier.question_id", question_id)
    w.add_string("classifier.type", "noul")
    w.add_tensor("classifier.weight", weight.astype(np.float32).reshape(n_embd))
    w.add_tensor("classifier.bias", np.array([bias], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


@pytest.fixture(scope="module")
def meta():
    s = tiny()
    s.start()
    res = s.make_request("GET", "/v1/models")
    s.stop()
    m = res.body["data"][0]["meta"]
    return {"n_embd": int(m["n_embd"]), "n_vocab": int(m["n_vocab"])}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = tiny()


def noul(res, question_id: str = "relevant") -> float:
    return res.body["answers"][question_id]["noul"]


def test_classify_constant_head(meta, tmp_path):
    # a zero weight leaves only the bias: p = sigmoid(bias) for any input
    head = write_head(str(tmp_path / "zero.gguf"), meta["n_embd"], 2, np.zeros(meta["n_embd"]), 1.5)
    server.extra_args = ["--classifier", head]
    server.start()
    expected = 1.0 / (1.0 + math.exp(-1.5))
    res = server.make_request("POST", "/classify", data={"input": "Once upon a time there was a cat."})
    assert res.status_code == 200
    assert abs(noul(res) - expected) < 1e-5
    res = server.make_request("POST", "/classify", data={"input": [1, 2, 3, 4, 5]})
    assert res.status_code == 200
    assert abs(noul(res) - expected) < 1e-5
    assert res.body["usage"]["input_tokens"] == 5


def test_classify_batch_matches_single_and_generation(meta, tmp_path):
    rng = np.random.default_rng(0)
    head = write_head(str(tmp_path / "rand.gguf"), meta["n_embd"], 3, rng.standard_normal(meta["n_embd"]) * 0.1, 0.0, "q")
    server.extra_args = ["--classifier", head]
    server.start()
    texts = ["Once upon a time there was a cat.", "The little girl went to the park to play."]
    single = [noul(server.make_request("POST", "/classify", data={"input": t}), "q") for t in texts]
    # generation on the same model must not change the classifier's answer
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 16})
    assert res.status_code == 200 and len(res.body["content"]) > 0
    batch = server.make_request("POST", "/classify", data={"input": texts})
    assert batch.status_code == 200
    results = batch.body if isinstance(batch.body, list) else batch.body["results"]
    assert len(results) == 2
    for r, s in zip(results, single):
        assert abs(r["answers"]["q"]["noul"] - s) < 1e-6
    assert noul(server.make_request("POST", "/classify", data={"input": texts[0]}), "q") == pytest.approx(single[0], abs=1e-6)


def test_classify_input_errors(meta, tmp_path):
    head = write_head(str(tmp_path / "zero.gguf"), meta["n_embd"], 2, np.zeros(meta["n_embd"]), 0.0)
    server.extra_args = ["--classifier", head, "--classifier-ctx", "32"]
    server.start()
    assert server.make_request("POST", "/classify", data={}).status_code == 400
    assert server.make_request("POST", "/classify", data={"input": ""}).status_code == 400
    assert server.make_request("POST", "/classify", data={"input": [meta["n_vocab"] + 5]}).status_code == 400
    res = server.make_request("POST", "/classify", data={"input": list(range(1, 100))})
    assert res.status_code == 400
    assert "--classifier-ctx" in res.body["error"]["message"]


def test_classify_without_head():
    server.start()
    assert server.make_request("POST", "/classify", data={"input": "hello"}).status_code == 501


def test_classify_head_layer_out_of_range(meta, tmp_path):
    head = write_head(str(tmp_path / "bad.gguf"), meta["n_embd"], 999, np.zeros(meta["n_embd"]), 0.0)
    server.extra_args = ["--classifier", head]
    with pytest.raises(Exception):
        server.start(timeout_seconds=30)


def test_classify_head_wrong_width(meta, tmp_path):
    head = write_head(str(tmp_path / "wide.gguf"), meta["n_embd"] + 1, 2, np.zeros(meta["n_embd"] + 1), 0.0)
    server.extra_args = ["--classifier", head]
    with pytest.raises(Exception):
        server.start(timeout_seconds=30)
