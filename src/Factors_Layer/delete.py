import os

from factor_config import FACTOR_DATA_DIR, FACTOR_RESULTS_DIR, PROJECT_ROOT


FACTOR_LAYER_DIRECTORIES = (
    FACTOR_DATA_DIR,
    FACTOR_RESULTS_DIR,
)


def validate_target(directory):
    project_root = os.path.realpath(PROJECT_ROOT)
    target = os.path.realpath(directory)

    if target == project_root:
        raise ValueError("Project root cannot be deleted")
    if os.path.commonpath((project_root, target)) != project_root:
        raise ValueError(f"Delete target is outside project: {target}")

    return target


def clear_directory(directory):
    target = validate_target(directory)

    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
        return

    for root, directories, files in os.walk(target, topdown=False):
        for name in files:
            if name == ".gitkeep":
                continue

            path = os.path.join(root, name)
            os.remove(path)
            print(f"Deleted: {path}")

        for name in directories:
            path = os.path.join(root, name)

            if not os.listdir(path):
                os.rmdir(path)
                print(f"Deleted empty directory: {path}")


def delete_factor_layer_data():
    for directory in FACTOR_LAYER_DIRECTORIES:
        clear_directory(directory)

    print("Factor Layer data and results deleted")


if __name__ == "__main__":
    delete_factor_layer_data()
