from datasets import mit_single_mouse_create_dataset, olympic_create_dataset, mit_single_mouse_label_maps, olympic_label_maps
from dpipe.factories import AugmentedDataset
from beartype import beartype

available_dataset = {'mitsinglemouse': mit_single_mouse_create_dataset,
                     'olympicsports': olympic_create_dataset}

available_label_maps = {'mitsinglemouse': mit_single_mouse_label_maps,
                        'olympicsports': olympic_label_maps}
@beartype
def get_label_map(name: str) -> dict:
    return available_label_maps[name]


@beartype
def create_dataset(name: str, *args, **kwargs) -> AugmentedDataset:
    name = name.lower()
    assert name in available_dataset.keys(), f"Dataset {name} is not available. Available: {list(available_dataset.keys())}"
    return available_dataset[name](*args, **kwargs)
