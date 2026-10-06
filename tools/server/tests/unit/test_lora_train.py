import math
import os
import time

import pytest
from utils import *

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


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = tiny()


def train_tokens(n: int) -> list[int]:
    # token ids inside the 512-token vocabulary of the test model, skipping the special ids 0-2
    return [3 + (i * 7) % 500 for i in range(n)]


def wait_for(job_id: int, timeout: float = 180.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        res = server.make_request("GET", "/lora/train")
        assert res.status_code == 200
        job = next(j for j in res.body if j["id"] == job_id)
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.5)
    raise TimeoutError(f"LoRA training job {job_id} did not finish")


def error_message(res) -> str:
    return res.body["error"]["message"]


def test_lora_train_disabled():
    server.start()
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600)})
    assert res.status_code == 501


def test_lora_train_job_runs_and_registers(tmp_path):
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)]
    server.start()
    # 600 tokens, n_ctx 256, stride 128 -> 3 windows -> 3 optimizer steps
    res = server.make_request("POST", "/lora/train", data={
        "tokens": train_tokens(600), "n_ctx": 256, "rank": 4, "epochs": 1, "lr": 1e-3, "file": "t.gguf",
    })
    assert res.status_code == 200, res.body
    assert res.body["n_steps"] == 3
    job = wait_for(res.body["id"])
    assert job["status"] == "done", job
    assert math.isfinite(job["loss_first"]) and math.isfinite(job["loss_last"])
    assert os.path.exists(tmp_path / "t.gguf")
    adapters = server.make_request("GET", "/lora-adapters").body
    assert any(a["path"].endswith("t.gguf") for a in adapters)
    # the server still generates afterwards
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8})
    assert res.status_code == 200


@pytest.mark.parametrize("args,body,flag", [
    ([],                                      {"rank": 200},           "--lora-train-max-rank"),
    ([],                                      {"n_ctx": 8192},         "--lora-train-max-ctx"),
    (["--lora-train-max-steps", "10"],        {"epochs": 20},          "--lora-train-max-steps"),
    (["--lora-train-max-tokens", "500"],      {},                      "--lora-train-max-tokens"),
])
def test_lora_train_limits(tmp_path, args, body, flag):
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)] + args
    server.start()
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256, **body})
    assert res.status_code == 400
    assert flag in error_message(res)


def test_lora_train_rejects_bad_requests(tmp_path):
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)]
    server.start()
    bad = [
        {"tokens": train_tokens(600), "n_ctx": 256, "file": "../escape.gguf"},   # outside --lora-train-dir
        {"tokens": train_tokens(100), "n_ctx": 256},                              # fewer than n_ctx + 1 tokens
        {"tokens": train_tokens(600), "n_ctx": 300},                              # not a multiple of 256
        {"tokens": [100000] * 600, "n_ctx": 256},                                 # outside the vocabulary
        {"n_ctx": 256},                                                           # no data
        {"tokens": train_tokens(600), "n_ctx": 256, "priority": "urgent"},
    ]
    for body in bad:
        res = server.make_request("POST", "/lora/train", data=body)
        assert res.status_code == 400, (body, res.body)


def test_lora_train_one_job_at_a_time_and_cancel(tmp_path):
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)]
    server.start()
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256, "epochs": 2000})
    assert res.status_code == 200
    job_id = res.body["id"]
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256})
    assert res.status_code == 400
    assert "already running" in error_message(res)
    res = server.make_request("POST", "/lora/train/cancel", data={"id": job_id})
    assert res.status_code == 200 and res.body["cancelled"] is True
    assert wait_for(job_id)["status"] == "cancelled"


def test_lora_train_needs_api_key(tmp_path):
    server.api_key = "secret"
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)]
    server.start()
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256})
    assert res.status_code == 401
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256},
                              headers={"Authorization": "Bearer secret"})
    assert res.status_code == 200
    wait_for_headers = {"Authorization": "Bearer secret"}
    server.make_request("POST", "/lora/train/cancel", data={"id": res.body["id"]}, headers=wait_for_headers)


def test_lora_train_rejects_moe(tmp_path):
    global server
    server = local_model(ServerPreset.stories15m_moe(), "TEST_MODEL_STORIES15M_MOE")
    server.extra_args = ["--lora-train", "--lora-train-dir", str(tmp_path)]
    server.start()
    res = server.make_request("POST", "/lora/train", data={"tokens": train_tokens(600), "n_ctx": 256})
    assert res.status_code == 400
    assert "MoE" in error_message(res)
