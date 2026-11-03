import os

from thairag.config import load_env


def test_load_env_reads_values_and_skips_comments(tmp_path, monkeypatch):
    monkeypatch.delenv("THAIRAG_TEST_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nTHAIRAG_TEST_KEY='abc'\n")
    load_env(env)
    assert os.environ["THAIRAG_TEST_KEY"] == "abc"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("THAIRAG_TEST_KEY", "from-shell")
    env = tmp_path / ".env"
    env.write_text("THAIRAG_TEST_KEY=from-file\n")
    load_env(env)
    assert os.environ["THAIRAG_TEST_KEY"] == "from-shell"


def test_load_env_missing_file_is_ok(tmp_path):
    load_env(tmp_path / "nope.env")
