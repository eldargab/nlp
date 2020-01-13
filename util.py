from collections import OrderedDict
import contextlib
import datetime as dt
import csv
import pandas as pd


def to_date(s):
    try:
        return dt.datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        return None


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


def frequencies(size_series: pd.Series):
    length = size_series.sum()
    size = size_series.sort_values(ascending=False)
    freq = size / length
    return pd.DataFrame.from_dict(OrderedDict([
        ('Size', size),
        ('Freq', freq),
        ('FreqCumSum', freq.cumsum()),
        ('SizeRevCumSum', size[::-1].cumsum()[::-1])
    ]))


def groups_summary(data: pd.DataFrame):
    sizes = data.groupby('Group').size()
    return frequencies(sizes)


def validate(predict, data):
    if isinstance(data, pd.DataFrame):
        g = data['Text'].map(predict)
        return pd.DataFrame(OrderedDict([
            ('Id', data['Id']),
            ('Date', data['Date']),
            ('Group', data['Group']),
            ('Guess', g)
        ]))
    elif isinstance(data, str):
        with contextlib.closing(select(data)) as lines:
            return _validate_lines(predict, lines)
    else:
        return _validate_lines(predict, data)


def _validate_lines(predict, data_lines):
    return pd.DataFrame(
        ((int(id), to_date(date), group, predict(txt)) for id, date, group, txt in data_lines),
        columns=['Id', 'Date', 'Group', 'Guess']
    )


def describe_results(r):
    hits = r.Group == r.Guess
    misses = ~hits

    stats = pd.DataFrame.from_dict(OrderedDict([
        ('Group', r.Group),
        ('Guess', r.Guess),
        ('Hit', hits),
        ('Miss', misses)
    ]))

    s = stats.groupby(['Group']).agg(
        Size=pd.NamedAgg('Hit', 'size'),
        Tp=pd.NamedAgg('Hit', 'sum')
    )

    fp = stats.groupby(['Guess']).agg(Fp=pd.NamedAgg('Miss', 'sum'))

    s = s.join(fp, how='outer')

    s['Precision'] = (s.Tp / (s.Tp + s.Fp)).map(lambda x: round(100 * x, ndigits=1))
    s['Recall'] = (s.Tp / s.Size).map(lambda x: round(100 * x, ndigits=1))
    s.sort_values(by='Size', ascending=False, inplace=True)

    s.fillna(0, inplace=True)
    s['Tp'] = s.Tp.astype(int)
    s['Fp'] = s.Fp.astype(int)

    return s
