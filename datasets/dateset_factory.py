from datasets import mit_single_mouse_create_dataset, olympic_create_dataset
from dpipe.factories import AugmentedDataset
from beartype import beartype

available_dataset = {'mitsinglemouse': mit_single_mouse_create_dataset,
                     'olympicsports': olympic_create_dataset}


@beartype
def create_dataset(name: str, *args, **kwargs) -> AugmentedDataset:
    name = name.lower()
    assert name in available_dataset.keys(), f"Dataset {name} is not available. Available: {list(available_dataset.keys())}"
    return available_dataset[name](*args, **kwargs)
