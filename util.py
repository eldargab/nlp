from collections import OrderedDict
import datetime as dt
import pandas as pd
import dask.dataframe as dd


def to_date(s):
    try:
        return dt.datetime.strptime(s, '%Y-%m-%d').date()
    except ValueError:
        return None


def read_csv_dataset(f: str) -> dd.DataFrame:
    return dd.read_csv(f, converters={'Date': to_date}, blocksize=None, sample=False)


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


def groups_summary(ds: pd.DataFrame):
    sizes = ds.groupby('Group').size()
    return frequencies(sizes)


def describe_results(r: pd.DataFrame):
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
