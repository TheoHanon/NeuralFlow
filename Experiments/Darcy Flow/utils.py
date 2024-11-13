from typing import Any, Tuple
import tensorflow as tf
import json


class DarcyDatasetLoader:

    def __init__(self, tfrecord_dir):
        self.tfrecord_dir = tfrecord_dir
        self.tfrecord_file = self.tfrecord_dir + "/data.tfrecord"
        self.metadata = self._load_metadata()

        self.image_shape = tuple(self.metadata["shape"])
        self.n_sample = self.metadata["n_sample"]
        self.full_dataset = self._load_full_dataset()

    # Feature description based on your dataset structure
    def _feature_description(self):
        """Feature description with dynamic image size."""

        flat_size  =   self.image_shape[0] * self.image_shape[1]

        print(flat_size)
        return {
            "permeability_field": tf.io.FixedLenFeature([flat_size], tf.float32),
            "solution_field": tf.io.FixedLenFeature([flat_size], tf.float32)
        }

    def _load_metadata(self):
        """Load metadata from JSON file."""
        with open(self.tfrecord_dir + "/_metadata.json", "r") as f:
            return json.load(f)

    def _parse_tfrecord(self, example_proto):
        """Parse each example in the TFRecord file."""
        example = tf.io.parse_single_example(example_proto, self._feature_description())
        
        # Reshape fields to 2D arrays using loaded image size
        permeability_field = tf.reshape(example["permeability_field"], self.image_shape)
        solution_field = tf.reshape(example["solution_field"], self.image_shape)
        
        return permeability_field, solution_field

    def _load_full_dataset(self):
        """Load the full dataset and apply parsing without batching or shuffling."""
        # Load and parse the dataset
        raw_dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        return raw_dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    def get_split(self, test_size=0.2, batch_size=32):
        """Split the dataset into training and validation sets upon request."""
        train_size = int((1 - test_size) * self.n_sample)
        
        # Shuffle the dataset without unbatching
        shuffled_dataset = self.full_dataset.shuffle(self.n_sample)
        
        # Split into train and validation sets, batch, and prefetch
        train_dataset = shuffled_dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        validation_dataset = shuffled_dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, validation_dataset
    
    def get_dataset(self, shuffle_buffer_size=100):
        """Return the full dataset with specified batch size and shuffle buffer size."""
        # No unbatching, just batching and shuffling directly
        return self.full_dataset.shuffle(shuffle_buffer_size).unbatch().prefetch(tf.data.AUTOTUNE)

            

