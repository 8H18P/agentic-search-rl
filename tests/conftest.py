"""Offline unit-suite boundary, including collection-time legacy client construction."""
import os
import socket

_previous_key = os.environ.get("LLM_API_KEY")
os.environ["LLM_API_KEY"] = "offline-unit-test-placeholder"
_original_connect = socket.socket.connect
_original_connect_ex = socket.socket.connect_ex
_original_create_connection = socket.create_connection


def _no_network(*args, **kwargs):
    raise RuntimeError("Network access is disabled in the offline unit test suite")


socket.socket.connect = _no_network
socket.socket.connect_ex = _no_network
socket.create_connection = _no_network


def pytest_unconfigure(config):
    socket.socket.connect = _original_connect
    socket.socket.connect_ex = _original_connect_ex
    socket.create_connection = _original_create_connection
    if _previous_key is None:
        os.environ.pop("LLM_API_KEY", None)
    else:
        os.environ["LLM_API_KEY"] = _previous_key
