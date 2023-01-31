import os
import re


def get_species_list(root_dir):
    species_list = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))]
    species_list.sort()
    return species_list


def snake_case(s):
    s = re.sub(r'[^\w\s]', '', s)  # remove punctuation
    s = re.sub('([A-Z]+)', r' \1', s)  # Separate all CAPS
    s = re.sub('([A-Z][a-z]+)', r' \1', s)  # Separate capitalized (CamelCase)
    return '_'.join(s.split()).lower()
