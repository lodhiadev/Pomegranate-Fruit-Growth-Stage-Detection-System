from pathlib import Path

def read_labels(label_path):
    objs = []
    with open(label_path) as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())
            objs.append((int(cls), w * h))
    return objs

def is_easy(objs):
    return all(area > 0.02 and cls != 0 for cls, area in objs)

def is_medium(objs):
    return any(area > 0.01 for _, area in objs)

def build_split(images_dir, labels_dir, out_txt, mode):
    images = sorted(Path(images_dir).glob("*.jpg"))
    selected = []

    for img in images:
        lbl = Path(labels_dir) / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        objs = read_labels(lbl)

        if mode == "easy" and is_easy(objs):
            selected.append(img)
        elif mode == "medium" and is_medium(objs):
            selected.append(img)
        elif mode == "hard":
            selected.append(img)

    with open(out_txt, "w") as f:
        for img in selected:
            f.write(str(img.resolve()) + "\n")

    print(f"{mode.upper()} → {len(selected)} images")
