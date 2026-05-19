from pathlib import Path


def fix_dataset_labels():
    input_dir = Path(
        r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\proposed_bboxes\vn3_2c\labels"
    )
    output_dir = Path(
        r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\proposed_bboxes\vn3_fixed\labels"
    )

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all txt files
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    fixed_files_count = 0
    total_files = len(txt_files)

    print(f"Processing {total_files} files...")

    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Check if the file has any bbox of the second class (class '1')
        has_second_class = any(line.strip().startswith("1 ") for line in lines)

        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if not parts:
                continue

            # Delete the second class bbox (the marker)
            if has_second_class and parts[0] == "1":
                continue

            # If any box is class '1', all remaining boxes become class '1'
            # Otherwise, they should all be class '0'
            if has_second_class:
                parts[0] = "1"
            else:
                parts[0] = "0"

            new_lines.append(" ".join(parts) + "\n")

        if has_second_class:
            fixed_files_count += 1

        output_file = output_dir / txt_file.name
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print(f"Processing complete. {total_files} files processed.")
    print(f"Files converted to class 1: {fixed_files_count}")
    print(f"Files kept as class 0: {total_files - fixed_files_count}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    fix_dataset_labels()
