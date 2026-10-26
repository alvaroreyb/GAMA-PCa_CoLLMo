"""Tests never call model services; CK integration is explicitly opt-in."""
import os

import pytest
import requests


def pytest_collection_modifyitems(items):
    for item in items:
        if '/integration/' in str(item.path):
            item.add_marker(pytest.mark.integration)
            if not os.environ.get('GAMA_PCA_KNOWLEDGE_DIR'):
                item.add_marker(pytest.mark.skip(reason='Set GAMA_PCA_KNOWLEDGE_DIR to the separate CK dataset'))


@pytest.fixture(autouse=True)
def isolated_outputs_and_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv('GAMA_PCA_OUTPUT_DIR', str(tmp_path / 'runtime'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Network/model calls are forbidden in tests')
    monkeypatch.setattr(requests.sessions.Session, 'request', forbidden)
