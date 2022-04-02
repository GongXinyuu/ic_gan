# -*- coding: utf-8 -*-
# @Date    : 4/2/22
# @Author  : Xinyu Gong (xinyu.gong@utexas.edu)
# @Link    : None
# @Version : 0.0

import os
import getpass
from tqdm import tqdm

user_name = getpass.getuser()

source_dataset_prefix = f"/home/{user_name}/datasets/imagenet_object_localization_challenge/ILSVRC/Data/CLS-LOC/train"
target_dataset_prefix = f"/home/{user_name}/datasets/imagenet_carni_train"

class_file_path = (
    f"/home/{user_name}/project/FUNIT/datasets/animals_train_class_names.txt"
)
with open(class_file_path, "r") as f:
    class_names = f.readlines()

os.makedirs(target_dataset_prefix, exist_ok=True)

for class_name in tqdm(class_names):
    source_dir = os.path.join(source_dataset_prefix, class_name.strip())
    target_dir = os.path.join(target_dataset_prefix, class_name.strip())
    assert os.path.exists(source_dir), f"{source_dir} doesn't exist"
    os.system(
        f"ln -s {source_dir} {target_dir}"
    )
