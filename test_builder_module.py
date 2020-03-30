from builder import task


@task
def a():
    print('a called')
    return 'a'


@task
def b():
    print('b called')
    return 'b'


@task
def ab():
    print('ab called')
    return a() + b()
