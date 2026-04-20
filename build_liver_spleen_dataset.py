import csv
from pathlib import Path


IMAGES_ROOT = Path("data/images")
LIVER_MASKS_ROOT = Path("data/masks_liver")
SPLEEN_MASKS_ROOT = Path("data/masks_spleen")
OUTPUT_CSV = Path("artifacts/liver_spleen_dataset.csv")
CLASS_TO_LABEL = {"normal": 0, "fatty_liver": 1}


def collect_samples():
    rows = []
    for class_name, label in CLASS_TO_LABEL.items():
        image_dir = IMAGES_ROOT / class_name
        liver_dir = LIVER_MASKS_ROOT / class_name
        spleen_dir = SPLEEN_MASKS_ROOT / class_name

        image_lookup = {p.stem.lower(): p for p in image_dir.iterdir() if p.is_file()}
        liver_lookup = {p.stem.lower(): p for p in liver_dir.iterdir() if p.is_file()}
        spleen_lookup = {p.stem.lower(): p for p in spleen_dir.iterdir() if p.is_file()}

        common_keys = sorted(image_lookup.keys() & liver_lookup.keys() & spleen_lookup.keys())
        missing_count = len(image_lookup.keys() - set(common_keys))
        print(f"{class_name}: matched={len(common_keys)} missing_any_mask={missing_count}")

        for key in common_keys:
            img_path = image_lookup[key]
            rows.append(
                {
                    "image_id": img_path.stem,
                    "class_name": class_name,
                    "label": label,
                    "image_path": str(img_path),
                    "liver_mask_path": str(liver_lookup[key]),
                    "spleen_mask_path": str(spleen_lookup[key]),
                }
            )
    return rows


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_samples()
    if not rows:
        raise RuntimeError("No matched image+liver+spleen samples were found.")

    fieldnames = ["image_id", "class_name", "label", "image_path", "liver_mask_path", "spleen_mask_path"]
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} matched samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
