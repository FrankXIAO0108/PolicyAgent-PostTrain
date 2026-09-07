import json
from unittest.mock import Mock

import pytest

from src.training.run_teacher_sft import load_sft_tokenizer, chat_tool_schemas


def test_legacy_tokens_preserve_ids(tmp_path):
    config = tmp_path / 'tokenizer_config.json'
    config.write_text(json.dumps({'extra_special_tokens': ['<tool>']}))
    inventory = tmp_path / 'tokenizer.json'
    inventory.write_text(json.dumps({'added_tokens': [{'content': '<tool>', 'id': 12}]}))
    before = config.read_bytes(), inventory.read_bytes()
    factory = Mock()
    factory.from_pretrained.return_value.convert_tokens_to_ids.return_value = 12
    load_sft_tokenizer(tmp_path, factory)
    factory.from_pretrained.assert_called_once_with(tmp_path, extra_special_tokens={})
    assert before == (config.read_bytes(), inventory.read_bytes())
    factory.from_pretrained.return_value.convert_tokens_to_ids.return_value = 13
    with pytest.raises(ValueError, match='changed saved token IDs'):
        load_sft_tokenizer(tmp_path, factory)


def test_missing_legacy_token_rejected(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'extra_special_tokens': ['missing']}))
    (tmp_path / 'tokenizer.json').write_text(json.dumps({'added_tokens': []}))
    factory = Mock()
    with pytest.raises(ValueError, match='missing from tokenizer.json'):
        load_sft_tokenizer(tmp_path, factory)
    factory.from_pretrained.assert_not_called()


def test_modern_metadata_not_overridden(tmp_path):
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'extra_special_tokens': {'tool_token': '<tool>'}}))
    factory = Mock()
    load_sft_tokenizer(tmp_path, factory)
    factory.from_pretrained.assert_called_once_with(tmp_path)


def test_bound_method_schema():
    pytest.importorskip('transformers')

    class Tool:
        def lookup(self, item: str) -> str:
            """Look up an item.

            Args:
                item: Item identifier.
            """
            return item

    schemas = chat_tool_schemas([Tool().lookup])
    assert schemas[0]['function']['name'] == 'lookup'
    assert set(schemas[0]['function']['parameters']['properties']) == {'item'}
    assert chat_tool_schemas(schemas) == schemas
