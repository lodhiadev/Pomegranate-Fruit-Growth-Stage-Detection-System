# utils/curriculum_split.py

from pathlib import Path
import random

def create_curriculum_splits(train_list, output_dir):
    """
    Splits training images into easy, medium, hard curriculum phases.
    """

    random.seed(42)

    train_list = list(train_list)
    random.shuffle(train_list)

    total = len(train_list)
    easy_end = int(0.4 * total)
    medium_end = int(0.7 * total)

    splits = {
        "easy": train_list[:easy_end],
        "medium": train_list[easy_end:medium_end],
        "hard": train_list[medium_end:]
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for phase, files in splits.items():
        out_file = output_dir / f"train_{phase}.txt"
        with open(out_file, "w") as f:
            for img in files:
                f.write(str(img) + "\n")

    return splits
