import json
import os
import datasets
from datasets import Features, Value, Image as ImageFeature, GeneratorBasedBuilder

class CustomDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for the custom dataset."""
    def __init__(self, jsonl_file=None, masks_folder=None, images_folder=None, **kwargs):
        super().__init__(**kwargs)
        self.jsonl_file = jsonl_file
        self.masks_folder = masks_folder
        self.images_folder = images_folder

class CustomDataset(GeneratorBasedBuilder):
    """A custom dataset for ControlNet training from a JSONL file and image folders."""

    # BUILDER_CONFIGS = [
    #     CustomDatasetConfig(
    #         name="default",
    #         version=datasets.Version("1.0.0"),
    #         description="Custom dataset from JSONL",
    #     ),
    # ]
    BUILDER_CONFIG_CLASS = CustomDatasetConfig
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Custom dataset with images, masks, and prompts.",
            features=Features({
                "image": ImageFeature(),
                "text": Value("string"),
                "conditioning_image": ImageFeature(),
            }),
            homepage="https://example.com/",
        )

    def _split_generators(self, dl_manager):
        # Data is local, so dl_manager is not strictly needed but is part of the API.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "jsonl_file": self.config.jsonl_file,
                    "masks_folder": self.config.masks_folder,
                    "images_folder": self.config.images_folder,
                },
            )
        ]

    def _generate_examples(self, jsonl_file, masks_folder, images_folder):
        """Yields examples."""
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for key, line in enumerate(f):
                data = json.loads(line)
                image_path = os.path.join(images_folder, data["image"])
                mask_path = os.path.join(masks_folder, data["mask"])

                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue  # Skip if files are missing

                yield key, {
                    "image": image_path,
                    "text": data["detail_prompt"],
                    "conditioning_image": mask_path,
                }

