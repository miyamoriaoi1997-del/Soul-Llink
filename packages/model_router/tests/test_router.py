#!/usr/bin/env python3
import json
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import Handler, RouterConfig, decide_route, extract_text_for_routing, make_request_hash


@pytest.fixture
def router_config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "listen": {"host": "127.0.0.1", "port": 18080},
                "audit": {"path": str(tmp_path / "audit.jsonl")},
                "upstream": {
                    "base_url": "https://upstream.example/v1",
                    "api_key": "test-key",
                    "timeout_seconds": 180,
                },
                "routing": {
                    "enabled": True,
                    "virtual_models": ["persona-auto", "persona-auto-technical", "persona-auto-sex"],
                    "default_model": "daily-model",
                    "work_model": "work-model",
                    "sex_model": "sex-model",
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def cfg(router_config_path):
    return RouterConfig(router_config_path)


def test_extract_text_string_and_list():
    payload = {"messages": [{"content": "abc"}, {"content": [{"type": "text", "text": "def"}]}]}
    assert extract_text_for_routing(payload) == "abc\ndef"


def test_persona_auto_default_routes_default(cfg):
    c = cfg
    d = decide_route({"model": "persona-auto", "messages": [{"role": "user", "content": "你好"}]}, c)
    assert d.route == "default"
    assert d.selected_model == c.routing["default_model"]


def test_keyword_without_state_machine_metadata_routes_default(cfg):
    d = decide_route({"model": "persona-auto", "messages": [{"role": "user", "content": "TDD pytest gateway 测试"}]}, cfg)
    assert d.route == "default"
    assert d.selected_model == cfg.routing["default_model"]
    assert d.reason == "virtual_default"


def test_state_machine_task_route_overrides_default_text(cfg):
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {"hermes_route_bucket": "task", "hermes_model_hint": "technical"},
        },
        cfg,
    )
    assert d.route == "technical"
    assert d.selected_model == cfg.routing["work_model"]
    assert d.reason == "state_machine:task"



def test_work_route_uses_work_model(cfg):
    d = decide_route({
        "model": "persona-auto",
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {"hermes_route_bucket": "work", "hermes_model_hint": "work"},
    }, cfg)

    assert d.route == "technical"
    assert d.selected_model == "work-model"
    assert d.reason == "state_machine:task"


def test_default_model_is_required_instead_of_code_model_fallback(router_config_path):
    data = yaml.safe_load(router_config_path.read_text(encoding="utf-8"))
    data["routing"].pop("default_model")
    router_config_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    cfg = RouterConfig(router_config_path)

    with pytest.raises(RuntimeError, match="routing.default_model is required"):
        decide_route({"model": "persona-auto", "messages": [{"role": "user", "content": "你好"}]}, cfg)

def test_state_machine_relationship_route_overrides_technical_keywords(cfg):
    c = cfg
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "pytest gateway"}],
            "metadata": {"hermes_route_bucket": "relationship", "hermes_model_hint": "default"},
        },
        c,
    )
    assert d.route == "default"
    assert d.selected_model == c.routing["default_model"]
    assert d.reason == "state_machine:relationship"


def test_state_machine_sex_route_uses_state_machine_signal_without_switch_allowed_gate(cfg):
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {"hermes_route_bucket": "sex", "hermes_model_hint": "sex"},
        },
        cfg,
    )
    assert d.route == "sex"
    assert d.selected_model == cfg.routing["sex_model"]
    assert d.reason == "state_machine:sex"

    d_false_flag = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {
                "hermes_route_bucket": "sex",
                "hermes_model_hint": "sex",
                "hermes_switch_allowed": False,
            },
        },
        cfg,
    )
    assert d_false_flag.route == "sex"
    assert d_false_flag.selected_model == cfg.routing["sex_model"]
    assert d_false_flag.reason == "state_machine:sex"


def test_state_machine_sex_route_overrides_technical_history_keywords(cfg):
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [
                {"role": "system", "content": "historical task context: TDD pytest gateway runtime Hermes"},
                {"role": "user", "content": "想要你。"},
            ],
            "metadata": {"hermes_route_bucket": "sex", "hermes_model_hint": "sex"},
        },
        cfg,
    )
    assert d.route == "sex"
    assert d.selected_model == cfg.routing["sex_model"]
    assert d.reason == "state_machine:sex"


def test_explicit_virtual_models_do_not_bypass_state_machine_metadata(cfg):
    c = cfg
    d_technical = decide_route({"model": "persona-auto-technical", "messages": [{"role": "user", "content": "你好"}]}, c)
    assert d_technical.route == "default"
    assert d_technical.selected_model == c.routing["default_model"]
    assert d_technical.reason == "virtual_default"

    d_sex = decide_route({"model": "persona-auto-sex", "messages": [{"role": "user", "content": "你好"}]}, c)
    assert d_sex.route == "default"
    assert d_sex.selected_model == c.routing["default_model"]
    assert d_sex.reason == "virtual_default"


def test_explicit_selected_virtual_model_falls_back_to_default(cfg):
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {"hermes_selected_model": "persona-auto"},
        },
        cfg,
    )
    assert d.route == "default"
    assert d.selected_model == cfg.routing["default_model"]
    assert d.reason == "virtual_default"


def test_explicit_technical_virtual_routes_technical(cfg):
    d = decide_route({
        "model": "persona-auto-technical",
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {"hermes_route_bucket": "task", "hermes_model_hint": "technical"},
    }, cfg)
    assert d.route == "technical"
    assert d.selected_model == cfg.routing["work_model"]
    assert d.reason == "state_machine:task"


def test_explicit_sex_virtual_routes_sex(cfg):
    d = decide_route({
        "model": "persona-auto-sex",
        "messages": [{"role": "user", "content": "你好"}],
        "metadata": {"hermes_route_bucket": "sex", "hermes_model_hint": "sex"},
    }, cfg)
    assert d.route == "sex"
    assert d.selected_model == cfg.routing["sex_model"]
    assert d.reason == "state_machine:sex"


def test_state_machine_selected_model_takes_priority_over_bucket(cfg):
    d = decide_route(
        {
            "model": "persona-auto",
            "messages": [{"role": "user", "content": "你好"}],
            "metadata": {
                "hermes_route_bucket": "relationship",
                "hermes_model_hint": "default",
                "hermes_selected_model": "glm-5-turbo",
            },
        },
        cfg,
    )
    assert d.route == "selected"
    assert d.selected_model == "glm-5-turbo"
    assert d.reason == "state_machine:selected_model"


def test_explicit_real_model_passthrough(cfg):
    d = decide_route({"model": "gpt-5.5", "messages": [{"role": "user", "content": "你好"}]}, cfg)
    assert d.route == "explicit"
    assert d.selected_model == "gpt-5.5"


def test_proxy_config_pins_real_upstream_after_hermes_points_to_proxy(cfg):
    c = cfg
    upstream = c.upstream()
    assert upstream.base_url == "https://upstream.example/v1"
    assert not hasattr(upstream, "default_model")


@pytest.mark.parametrize("base_url", [
    "file:///etc/passwd",
    "ftp://example.com/v1",
    "http://user:password@example.com/v1",
    "http://example.com/v1?redirect=file:///etc/passwd",
    "http://example.com:bad/v1",
    "http://example.com:99999/v1",
])
def test_upstream_rejects_unsafe_url_shapes(router_config_path, base_url):
    data = yaml.safe_load(router_config_path.read_text(encoding="utf-8"))
    data["upstream"]["base_url"] = base_url
    router_config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(RuntimeError, match="upstream base_url"):
        RouterConfig(router_config_path).upstream()


def test_request_hash_does_not_expose_prompt():
    raw = json.dumps({"messages": [{"content": "secret prompt text"}]}, ensure_ascii=False).encode()
    h = make_request_hash(raw)
    assert len(h) == 16
    assert "secret" not in h
    assert "prompt" not in h


def test_health_payload_supports_health_alias_and_reports_models(cfg):
    handler = object.__new__(Handler)
    handler.server = type("Server", (), {"cfg": cfg})()

    payload = handler._health_payload()

    assert payload["ok"] is True
    assert payload["upstream_base_url"] == "https://upstream.example/v1"
    assert payload["default_model"] == "daily-model"
    assert payload["work_model"] == "work-model"
    assert "technical_model" not in payload
    assert payload["sex_model"] == "sex-model"
    assert payload["routing_enabled"] is True


def test_v1_models_payload_exposes_virtual_and_backing_models(cfg):
    handler = object.__new__(Handler)
    handler.server = type("Server", (), {"cfg": cfg})()

    payload = handler._models_payload()
    ids = [item["id"] for item in payload["data"]]

    assert payload["object"] == "list"
    assert "persona-auto" in ids
    assert "persona-auto-technical" in ids
    assert "persona-auto-sex" in ids
    assert "work-model" in ids
    assert "sex-model" in ids
    assert len(ids) == len(set(ids))
