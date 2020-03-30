from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any, Mapping, TypeVar, NamedTuple, Union, MutableMapping
import os


T = TypeVar("T")
Id = str


class Task(NamedTuple):
    id: Id
    f: Any


def get_task(f: Any) -> Task:
    try:
        return f._builder_task
    except AttributeError:
        raise TypeError(f'{f} is not a builder task')


class Rec:
    id: Id
    f: Any
    value: Any
    deps: MutableMapping[Id, Any]
    epoch: int

    def __init__(self, task):
        self.id = task.id
        self.f = task.f
        self.value = None
        self.deps = {}
        self.epoch = 0


class Builder:
    _module: Mapping[Id, Any]
    _temp_dir: str
    _state: MutableMapping[Id, Rec]
    _current_task: Union[Rec, None]
    _epoch: int


    def __init__(self, module, temp_dir='tmp'):
        self._module = module if isinstance(module, dict) else vars(module)
        self._temp_dir = os.path.abspath(temp_dir)
        self._state = {}
        self._epoch = 1
        self._current_task = None


    @property
    def temp_dir(self):
        return self._temp_dir


    @contextmanager
    def _new_task_ctx(self, rec: Union[Rec, None]):
        prev = self._current_task
        self._current_task = rec
        try:
            yield rec
        finally:
            self._current_task = prev


    def _eval(self, task_id: Id):
        wrapped_fn = self._module.get(task_id)
        if not wrapped_fn:
            raise ValueError(f'{task_id} is not defined in the current builder')

        t = get_task(wrapped_fn)
        rec = self._state.get(task_id)

        if not rec or rec.f != t.f:
            rec = Rec(t)
            self._state[rec.id] = rec

        self._compute(rec)

        if self._current_task:
            self._current_task.deps[rec.id] = rec.value

        return rec.value


    def _compute(self, rec: Rec):
        if rec.epoch == self._epoch:
            return

        if rec.epoch == 0 or self._has_changed_deps(rec):
            rec.deps.clear()
            with self._new_task_ctx(rec):
                rec.value = rec.f()

        rec.epoch = self._epoch


    def _has_changed_deps(self, rec: Rec):
        with self._new_task_ctx(None):
            for dep, old_val in rec.deps.items():
                if old_val != self._eval(dep):
                    return True
        return False


    def run(self, f: Callable[[], T]) -> T:
        t = get_task(f)
        self._epoch += 1
        global _BUILDER
        prev_builder = _BUILDER
        _BUILDER = self
        try:
            return self._eval(t.id)
        finally:
            _BUILDER = prev_builder


_BUILDER = None


def task(f):
    t = Task(f.__name__, f)

    @wraps(f)
    def task_fn():
        global _BUILDER
        if _BUILDER:
            return _BUILDER._eval(t.id)
        else:
            return f()

    task_fn._builder_task = t

    return task_fn


class AlreadyExists(Exception):
    pass


_FILE = None


def _register_file(filename):
    global _FILE
    _FILE = filename
    if os.path.exists(filename):
        raise AlreadyExists


def file_task(f: Callable[[], None]) -> Callable[[], str]:

    @wraps(f)
    def file_task_fn():
        global _FILE
        prev_file = _FILE
        _FILE = None
        try:
            f()
            if not _FILE:
                raise RuntimeError(f'build_filename() was not called within a file task {f.__name__}')
            return _FILE
        except AlreadyExists:
            assert _FILE
            return _FILE
        finally:
            _FILE = prev_file

    return task(file_task_fn)


def build_filename(name: str) -> str:
    global _BUILDER
    temp_dir = _BUILDER.temp_dir if _BUILDER else os.path.abspath('tmp')
    file = os.path.join(temp_dir, name)
    _register_file(file)
    return file


__all__ = [
    'Builder',
    'task',
    'file_task',
    'build_filename'
]
