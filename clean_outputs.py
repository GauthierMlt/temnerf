import glob
import os
import shutil

if __name__ == "__main__":

    dirs = glob.glob("outputs/*/*/slices")

    for slices_dir in dirs:
        # Check if the "slices" directory is empty
        if not os.listdir(slices_dir):
            experiment_dir = os.path.dirname(slices_dir)
            print(f"Removing failed experiment directory: {experiment_dir}")
            shutil.rmtree(experiment_dir)