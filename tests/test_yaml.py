import pytest
from pathlib import Path
from simple_backtester.utils import yaml_helper


@pytest.fixture
def expected_yaml_file():
    return Path("tests/test_data/expected_yaml.yaml").read_text().replace(" ", "")


def test_yaml_dump(tmp_path, expected_yaml_file):
    file_path = tmp_path / "test.yaml"
    data = {
        "level1": {
            "level2": {
                "item1": [1, 2, 3, 4, 5],
                "item2": {"subitem1": "value1", "subitem2": "value2"},
                "item3": [
                    {"key1": "value1", "key2": "value2"},
                    {"key3": "value3", "key4": "value4"},
                ],
            },
            "level2b": {
                "list1": [i for i in range(3)],
                "nested_dict": {
                    f"key_{i}": {f"nested_key_{j}": j for j in range(2)}
                    for i in range(2)
                },
            },
        },
        "large_list": [{f"entry_{i}": i for i in range(5)} for _ in range(3)],
    }
    yaml_helper.YamlParser(file_path).save_yaml(data)
    assert file_path.read_text().replace(" ", "") == expected_yaml_file
