"""Catalogue expansion must stay bounded and invalid captures stay out of search."""
import json

import pytest

from tonehound import library_manager as library
from tonehound import native_bridge as bridge
from tonehound.tone3000 import Model, Tone, Tone3000Error


class Catalogue:
    configured = True
    connected = True

    def __init__(self):
        self.downloads = []
        self.queries = []
        self.invalid = False
        self.rate_limit = False

    def get_tone(self, tone_id):
        return Tone.from_json({'id': tone_id, 'title': 'Creator title',
                              'description': '<p>Original creator description &amp; settings</p>',
                              'user': {'username': 'creator'}, 'makes': [{'name': 'Acme Amp 100'}]})

    def list_models(self, tone_id, **kwargs):
        return [Model(100 * tone_id + i, tone_id, f'Capture {i}', 'https://example.invalid/model.nam',
                      architecture='1') for i in range(1, 10)]

    def download_model(self, model, directory):
        if self.rate_limit:
            raise Tone3000Error('rate_limited', 'Wait a minute')
        self.downloads.append(model.id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f't3k-{model.tone_id}-{model.id} Capture.nam'
        data = {'architecture': 'Linear', 'config': {'receptive_field': 3, 'bias': True},
                'weights': [.2, -.5, .9, .01]}
        path.write_text('not a NAM' if self.invalid else json.dumps(data), encoding='utf-8')
        return path

    def search_tones(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return {'data': [self.get_tone(42).raw], 'total_pages': 9}


def test_browsing_is_one_bounded_page_without_downloads(tmp_path):
    client = Catalogue()
    result = library.search(tmp_path, client, {'query': 'Amp', 'page': 3, 'page_size': 12})
    assert result['page'] == 3 and result['total_pages'] == 9
    assert client.queries[0][1]['architecture'] == 1
    assert client.queries[0][1]['page_size'] == 12
    assert client.downloads == []
    assert result['tones'][0]['description'] == 'Original creator description & settings'
    assert result['tones'][0]['makes'] == ['Acme Amp 100']
    assert result['tones'][0]['models'] == []  # Never infer physical gear from a capture title.


@pytest.mark.parametrize('payload', [{'tone_ids': []}, {'tone_ids': list(range(1, 7))},
    {'tone_ids': [1], 'max_models': 26}, {'tone_ids': [1], 'models_per_tone': 6}, {'tone_ids': [True]}])
def test_fetch_limits_fail_before_network(tmp_path, payload):
    client = Catalogue()
    with pytest.raises(ValueError):
        library.fetch(tmp_path, client, payload, lambda *_: None)
    assert client.downloads == []


def test_expansion_adds_only_requested_number_and_later_batches_skip_existing(tmp_path):
    client = Catalogue()
    request = {'tone_ids': [1, 2], 'models_per_tone': 2, 'max_models': 3}
    result = library.fetch(tmp_path, client, request, lambda *_: None)
    assert [r['model_id'] for r in result['added']] == [101, 102, 201]
    assert result['profile_count'] == 3
    assert len(list((tmp_path / '.cache/tone3000/profiles').glob('*.nam'))) == 3
    second = library.fetch(tmp_path, client, {'tone_ids': [1], 'max_models': 1}, lambda *_: None)
    assert [r['model_id'] for r in second['added']] == [103]
    assert [r['model_id'] for r in second['skipped']] == [101, 102]
    row = bridge.library(tmp_path)['profiles'][0]
    assert row['description'] == 'Original creator description & settings'
    assert row['makes'] == ['Acme Amp 100']
    assert not list((tmp_path / '.cache/tone3000/incoming').glob('*.nam'))


def test_invalid_model_does_not_poison_local_index(tmp_path):
    client = Catalogue()
    client.invalid = True
    result = library.fetch(tmp_path, client, {'tone_ids': [1], 'max_models': 1}, lambda *_: None)
    assert not result['added'] and len(result['errors']) == 1
    assert result['profile_count'] == 0
    assert not list((tmp_path / '.cache/tone3000/incoming').glob('*.nam'))


def test_rate_limit_stops_batch_immediately(tmp_path):
    client = Catalogue()
    client.rate_limit = True
    with pytest.raises(Tone3000Error, match='Wait a minute'):
        library.fetch(tmp_path, client, {'tone_ids': [1, 2]}, lambda *_: None)
    assert not client.downloads


def test_status_contains_no_credentials(tmp_path):
    client = Catalogue()
    client.key = 'private-fixture-key'
    client.token = 'private-fixture-token'
    result = library.status(tmp_path, client)
    encoded = json.dumps(result)
    assert 'private-fixture' not in encoded
    assert result['profile_count'] == 0 and result['connected']


def test_search_rejects_api_page_overflow(tmp_path):
    with pytest.raises(ValueError):
        library.search(tmp_path, Catalogue(), {'page_size': 26})
