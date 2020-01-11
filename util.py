import csv


def select(csv_file, cols=None, skip_first=True):
    with open(csv_file) as lines:
        skipped = not skip_first
        for line in csv.reader(lines, delimiter=',', quotechar='"'):
            if not skipped:
                skipped = True
                continue
            if cols:
                line = tuple(map(lambda i: line[i], cols))
            yield line


def select_group_text_pairs(sample_file):
    return select(sample_file, (2, 3))
