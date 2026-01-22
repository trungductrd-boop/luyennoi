import pytest

from server.audio_api import app


def test_vocab_id_int_and_string_parity():
    client = app.test_client()
    # Post vocab_id as string
    resp_str = client.post('/compare', data={'vocab_id': '1'})
    # Post vocab_id as int-like value (Flask will convert form values to strings,
    # this checks server treats both the same way)
    resp_int = client.post('/compare', data={'vocab_id': 1})

    assert resp_str.status_code == resp_int.status_code
    # For unknown ids the API returns 400 with a structured error
    assert resp_str.status_code == 400
    j = resp_str.get_json()
    assert isinstance(j, dict)
    assert j.get('success') is False
    assert j.get('error') == 'unknown vocab_id'
    assert 'vocab_id' in j
