from unittest import TestCase
from .dateset_factory import create_dataset


class TestDatasetFactory(TestCase):
    def test_create_mit_mouse(self):
        name = 'mitsinglemouse'
        dataset_path = './data/mouse'
        dataset = create_dataset(name, dataset_path=dataset_path, with_labels=False, shuffle=False)
        self.assertEqual(dataset.length, 6)

    def test_create_mit_mouse_worng_path(self):
        name = 'mitsinglemouse'
        dataset_path = './data/human'
        with self.assertRaises(Exception) as context:
            dataset = create_dataset(name, dataset_path=dataset_path, with_labels=False, shuffle=False)
        self.assertTrue('list index out of range' in str(context.exception))

    def test_create_olympic_sports(self):
        name = 'olympicsports'
        dataset_path = './data/human'
        dataset = create_dataset(name, dataset_path=dataset_path, with_labels=False, shuffle=False)
        self.assertEqual(dataset.length, 2)

    def test_create_olympic_sports_wrong_path(self):
        name = 'olympicsports'
        dataset_path = './data/mouse'
        with self.assertRaises(Exception) as context:
            create_dataset(name, dataset_path=dataset_path, with_labels=False, shuffle=False)
        self.assertTrue('list index out of range' in str(context.exception))
