import hashlib
from simple_backtester.utils import yaml_helper


def test_yaml_dump(tmp_path):
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
    assert (
        hashlib.md5(file_path.read_text().replace(" ", "").encode()).hexdigest()
        == "4225f62ca43faaa0a8476aed0af80698"
    )
