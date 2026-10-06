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


def write_head(path: str, n_embd: int, layer: int, weight: np.ndarray, bias, question_id: str = "relevant",
               type: str = "noul", pooling: str = "mean", options: list[str] | None = None) -> str:
    """weight: n_embd values (noul, score) or K x n_embd (choice); bias: a number (noul), K values (choice) or K-1 thresholds (score)"""
    w = gguf.GGUFWriter(path, "classifier")
    w.add_type("classifier")
    w.add_uint32("classifier.layer", layer)
    w.add_string("classifier.question_id", question_id)
    w.add_string("classifier.type", type)
    w.add_string("classifier.pooling", pooling)
    if options is not None:
        w.add_array("classifier.options", options)
    weight = np.asarray(weight, dtype=np.float32)
    w.add_tensor("classifier.weight", weight if weight.ndim == 2 else weight.reshape(n_embd))
    w.add_tensor("classifier.bias", np.atleast_1d(np.asarray(bias, dtype=np.float32)))
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


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def test_classify_choice_constant(meta, tmp_path):
    # zero weights: the probabilities are softmax(bias)
    bias = np.array([0.5, 2.0, -1.0])
    head = write_head(str(tmp_path / "choice.gguf"), meta["n_embd"], 2, np.zeros((3, meta["n_embd"])), bias,
                      "topic", type="choice", options=["cats", "dogs", "birds"])
    server.extra_args = ["--classifier", head]
    server.start()
    res = server.make_request("POST", "/classify", data={"input": "Once upon a time there was a cat."})
    assert res.status_code == 200
    a = res.body["answers"]["topic"]
    p = np.exp(bias) / np.exp(bias).sum()
    assert a["type"] == "choice" and a["choice"] == "dogs"
    assert a["confidence"] == pytest.approx(p[1], abs=1e-6)
    for opt, q in zip(["cats", "dogs", "birds"], p):
        assert a["probabilities"][opt] == pytest.approx(q, abs=1e-6)


def test_classify_score_constant(meta, tmp_path):
    # zero weight: P(y <= k) = sigmoid(theta_k)
    theta = np.array([-1.0, 0.0, 1.5])
    legend = ["none", "low", "medium", "high"]
    head = write_head(str(tmp_path / "score.gguf"), meta["n_embd"], 2, np.zeros(meta["n_embd"]), theta,
                      "severity", type="score", options=legend)
    server.extra_args = ["--classifier", head]
    server.start()
    res = server.make_request("POST", "/classify", data={"input": [1, 2, 3]})
    assert res.status_code == 200
    a = res.body["answers"]["severity"]
    cum = np.concatenate([sigmoid(theta), [1.0]])
    p = np.diff(np.concatenate([[0.0], cum]))
    assert a["type"] == "score"
    assert a["score"] == pytest.approx(float((np.arange(4) * p).sum()), abs=1e-6)
    assert a["confidence"] == pytest.approx(p.max(), abs=1e-6)
    for k in range(4):
        assert a["probabilities"][str(k)] == pytest.approx(p[k], abs=1e-6)
        assert a["legend"][str(k)] == legend[k]


def test_classify_score_bad_thresholds(meta, tmp_path):
    head = write_head(str(tmp_path / "bad.gguf"), meta["n_embd"], 2, np.zeros(meta["n_embd"]), [1.0, 0.0],
                      type="score", options=["a", "b", "c"])
    server.extra_args = ["--classifier", head]
    with pytest.raises(Exception):
        server.start(timeout_seconds=30)


def test_features_match_heads(meta, tmp_path):
    # the heads' answers must equal the same linear functions of the /features vectors
    rng = np.random.default_rng(1)
    n = meta["n_embd"]
    w1, w2, wc = rng.standard_normal(n) * 0.1, rng.standard_normal(n) * 0.1, rng.standard_normal((2, n)) * 0.1
    heads = [
        write_head(str(tmp_path / "m.gguf"), n, 3, w1, 0.2, "mean3"),
        write_head(str(tmp_path / "l.gguf"), n, 2, w2, -0.1, "last2", pooling="last"),
        write_head(str(tmp_path / "c.gguf"), n, 3, wc, [0.0, 0.3], "choice3", type="choice", pooling="last", options=["x", "y"]),
    ]
    server.extra_args = ["--features"] + sum([["--classifier", h] for h in heads], [])
    server.start()
    text = "The little girl went to the park to play."
    f = server.make_request("POST", "/features", data={"input": text, "layers": [2, 3]})
    assert f.status_code == 200
    feats = f.body["features"]
    assert set(feats.keys()) == {"2", "3"} and set(feats["3"].keys()) == {"mean", "last"}
    assert len(feats["3"]["mean"]) == n
    a = server.make_request("POST", "/classify", data={"input": text}).body["answers"]
    assert a["mean3"]["noul"] == pytest.approx(sigmoid(w1 @ np.array(feats["3"]["mean"]) + 0.2), abs=1e-5)
    assert a["last2"]["noul"] == pytest.approx(sigmoid(w2 @ np.array(feats["2"]["last"]) - 0.1), abs=1e-5)
    z = wc @ np.array(feats["3"]["last"]) + np.array([0.0, 0.3])
    p = np.exp(z - z.max()) / np.exp(z - z.max()).sum()
    assert a["choice3"]["probabilities"]["y"] == pytest.approx(p[1], abs=1e-5)
    # one pooling, default layer (the last one), a batch of token inputs
    b = server.make_request("POST", "/features", data={"input": [[1, 2, 3], [4, 5, 6, 7]], "pooling": "last"})
    assert b.status_code == 200 and len(b.body) == 2
    assert list(b.body[0]["features"].values())[0].keys() == {"last"}
    assert b.body[1]["n_tokens"] == 4
    # a layer outside the model
    assert server.make_request("POST", "/features", data={"input": text, "layers": [999]}).status_code == 400
    assert server.make_request("POST", "/features", data={"input": text, "pooling": "max"}).status_code == 400


def test_features_only(meta):
    # --features without heads: /features works, /classify reports no heads
    server.extra_args = ["--features"]
    server.start()
    assert server.make_request("POST", "/features", data={"input": "hello", "layers": 1}).status_code == 200
    assert server.make_request("POST", "/classify", data={"input": "hello"}).status_code == 501


def test_features_disabled():
    server.start()
    assert server.make_request("POST", "/features", data={"input": "hello"}).status_code == 501


def toy_data(n_per_class: int = 12):
    # three easily separated kinds of input for the 260K-parameter test model
    rng = np.random.default_rng(2)
    animals = ["cat", "dog", "bird"]
    data = []
    for k, a in enumerate(animals):
        for i in range(n_per_class):
            name = ["Tom", "Lily", "Ben", "Sue"][rng.integers(4)]
            data.append({"input": f"{name} saw a {a}. The {a} was big and the {a} liked {name}.", "label": a, "k": k})
    return data


def test_classify_train_choice_and_score(meta, tmp_path):
    server.extra_args = ["--classifier-train", "--classifier-dir", str(tmp_path)]
    server.start()
    data = toy_data()
    body = {"question_id": "animal", "type": "choice", "data": [{"input": d["input"], "label": d["label"]} for d in data],
            "layers": [2, 4], "C": [0.1, 1.0], "folds": 3, "save": True}
    res = server.make_request("POST", "/classify/train", data=body)
    assert res.status_code == 200, res.body
    r = res.body
    assert r["question_id"] == "animal" and r["type"] == "choice" and r["layer"] in (2, 4)
    assert r["options"] == ["bird", "cat", "dog"]
    assert len(r["grid"]) == 2 * 2 * 2
    assert r["cv"]["accuracy"] > 0.8
    assert os.path.exists(r["saved"])
    right = 0
    for d in data:
        a = server.make_request("POST", "/classify", data={"input": d["input"]}).body["answers"]["animal"]
        right += a["choice"] == d["label"]
    assert right / len(data) > 0.9

    # score: the level index as the label, a legend per level; registered next to the choice head
    body = {"question_id": "level", "type": "score", "options": ["cat", "dog", "bird"],
            "data": [{"input": d["input"], "label": d["k"]} for d in data], "layers": 3, "pooling": "mean", "folds": 3}
    res = server.make_request("POST", "/classify/train", data=body)
    assert res.status_code == 200, res.body
    assert res.body["layer"] == 3 and res.body["pooling"] == "mean" and res.body["saved"] is None
    assert {h["question_id"] for h in res.body["heads"]} == {"animal", "level"}
    a = server.make_request("POST", "/classify", data={"input": data[0]["input"]}).body["answers"]
    assert set(a.keys()) == {"animal", "level"}
    assert a["level"]["legend"] == {"0": "cat", "1": "dog", "2": "bird"}
    assert a["level"]["score"] < 0.8  # data[0] is a cat (level 0)

    # retraining replaces the head
    body = {"question_id": "animal", "type": "noul", "data": [{"input": d["input"], "label": d["label"] == "cat"} for d in data],
            "layers": [5], "folds": 3}
    res = server.make_request("POST", "/classify/train", data=body)
    assert res.status_code == 200, res.body
    assert len(res.body["heads"]) == 2
    a = server.make_request("POST", "/classify", data={"input": data[0]["input"]}).body["answers"]
    assert a["animal"]["type"] == "noul" and a["animal"]["noul"] > 0.5
    saved = str(tmp_path / "animal.gguf")
    server.stop()

    # the saved file is the choice head; a server started with it gives the same answers
    server.extra_args = ["--classifier", saved]
    server.start()
    a = server.make_request("POST", "/classify", data={"input": data[0]["input"]}).body["answers"]["animal"]
    assert a["type"] == "choice" and a["choice"] == "cat"


def test_classify_train_errors(meta, tmp_path):
    data = [{"input": d["input"], "label": d["label"]} for d in toy_data(4)]
    server.start()
    assert server.make_request("POST", "/classify/train", data={"question_id": "q", "type": "choice", "data": data}).status_code == 501
    server.stop()
    server.extra_args = ["--classifier-train"]
    server.start()
    ok = {"question_id": "q", "type": "choice", "data": data, "layers": 2, "folds": 3}
    assert server.make_request("POST", "/classify/train", data={**ok, "question_id": "../x"}).status_code == 400
    assert server.make_request("POST", "/classify/train", data={**ok, "folds": 5}).status_code == 400  # 4 items per class
    assert server.make_request("POST", "/classify/train", data={**ok, "save": True}).status_code == 400  # no --classifier-dir
    assert server.make_request("POST", "/classify/train", data={**ok, "type": "score"}).status_code == 400  # no options
    assert server.make_request("POST", "/classify/train", data={**ok, "type": "noul"}).status_code == 400  # labels not true/false
    assert server.make_request("POST", "/classify/train", data={**ok, "layers": 999}).status_code == 400
    assert server.make_request("POST", "/classify/train", data={**ok, "C": 0}).status_code == 400
    assert server.make_request("POST", "/classify/train", data=ok).status_code == 200
