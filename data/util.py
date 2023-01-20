import re


def snake_case(s):
    s = re.sub(r'[^\w\s]', '', s)  # remove punctuation
    s = re.sub('([A-Z]+)', r' \1', s)  # Separate all CAPS
    s = re.sub('([A-Z][a-z]+)', r' \1', s)  # Separate capitalized (CamelCase)
    return '_'.join(s.split()).lower()
