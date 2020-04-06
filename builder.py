import os
import struct
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any, TypeVar, NamedTuple, Union, MutableMapping, MutableSet, List, Iterable, Tuple
from zlib import crc32


T = TypeVar("T")
Id = str
Mtime = float
CRC = int


class Task(NamedTuple):
    id: Id
    f: Any


def get_task(f: Any) -> Task:
    try:
        return f._builder_task
    except AttributeError:
        raise TypeError(f'{f} is not a builder task')


def get_mtime(path) -> Mtime:
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0


def crc_int(val: int, crc: CRC) -> CRC:
    return crc32(struct.pack('n', val), crc)


def crc_float(val: float, crc: CRC) -> CRC:
    return crc32(struct.pack('f', val), crc)


def crc_str(val: str, crc: CRC) -> CRC:
    return crc32(val.encode(), crc)


def crc_files(files: Iterable[str], crc: CRC) -> CRC:
    for f in sorted(files):
        crc = crc_str(f, crc)
        crc = crc_float(get_mtime(f), crc)
    return crc


class Rec:
    id: Id
    f: Any
    value: Any
    deps: MutableMapping[Id, Tuple[Any, CRC]]
    src_files: MutableSet[str]
    output_files: List[str]
    crc: CRC
    epoch: int

    def __init__(self, task):
        self.id = task.id
        self.f = task.f
        self.value = None
        self.deps = OrderedDict()
        self.src_files = set()
        self.output_files = []
        self.crc = 0
        self.epoch = 0

    def reset(self):
        self.deps.clear()
        self.src_files.clear()
        self.output_files.clear()
        self.crc = 0

    def reg_dep(self, rec: 'Rec'):
        self.deps[rec.id] = (rec.value, rec.crc)
        if rec.crc > 0:
            self.crc = crc_int(rec.crc, self.crc)

    def compute(self):
        self.value = self.f()
        self.crc = self.complete_crc()

    def complete_crc(self) -> CRC:
        return crc_files(self.src_files, self.crc)



class Builder:
    _module: MutableMapping[Id, Any]  # TODO: probably we should not mutate passed modules
    _temp_dir: str
    _state: MutableMapping[Id, Rec]
    _current_task: Union[Rec, None]
    _epoch: int
    _running: bool


    def __init__(self, module, temp_dir='tmp'):
        self._module = module if isinstance(module, dict) else vars(module)
        self._temp_dir = os.path.abspath(temp_dir)
        self._state = {}
        self._epoch = 1
        self._current_task = None
        self._running = False


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


    def _eval(self, task_id: Id) -> Rec:
        wrapped_fn = self._module.get(task_id)
        if not wrapped_fn:
            raise ValueError(f'{task_id} is not defined in the current builder')

        t = get_task(wrapped_fn)
        rec = self._state.get(task_id)

        if not rec or rec.f is not t.f:
            rec = Rec(t)
            self._state[rec.id] = rec

        self._compute(rec)

        if self._current_task:
            self._current_task.reg_dep(rec)

        return rec


    def _compute(self, rec: Rec):
        if rec.epoch == self._epoch:
            return

        if rec.epoch == 0 or self._has_changed_deps(rec):
            rec.reset()
            with self._new_task_ctx(rec):
                rec.compute()

        rec.epoch = self._epoch


    def _has_changed_deps(self, rec: Rec):
        for f in rec.output_files:
            if not os.path.exists(f):
                return True

        crc = 0
        with self._new_task_ctx(None):
            for dep_id, (old_val, old_crc) in rec.deps.items():
                dep = self._eval(dep_id)
                if dep.value is not old_val:
                    return True
                if dep.crc != old_crc:
                    return True
                if dep.crc > 0:
                    crc = crc_int(dep.crc, crc)

        crc = crc_files(rec.src_files, crc)

        return crc != rec.crc


    def reg_src(self, f: str):
        if self._current_task:
            self._current_task.src_files.add(os.path.abspath(f))


    def set_constant(self, name: str, val):
        if self._current_task:
            raise RuntimeError('set_constant() is not allowed within a task')
        const = lambda: val
        const.__name__ = name
        self._module[name] = task(const)


    @contextmanager
    def session(self):
        self._epoch += 1
        self._running = True
        global _BUILDER
        prev_builder = _BUILDER
        _BUILDER = self
        try:
            yield
        finally:
            self._running = False
            _BUILDER = prev_builder


    def eval(self, task_id: str):
        if self._running:
            return self._eval(task_id).value
        else:
            with self.session():
                return self._eval(task_id).value


    def run(self, f: Callable[[], T]) -> T:
        try:
            t = task._builder_task
        except AttributeError:
            t = None

        if self._running:
            return self._eval(t.id).value if t else f()
        else:
            with self.session():
                return self._eval(t.id).value if t else f()


    def output(self, name: str, build_fn: Callable[[str], None]) -> str:
        if not self._current_task:
            raise RuntimeError('No current task')

        crc = self._current_task.complete_crc()
        path = os.path.abspath(os.path.join(self.temp_dir, name))

        if crc:
            prefix, ext = os.path.splitext(path)
            filename = prefix + '-' + '{:x}'.format(crc) + ext
        else:
            filename = path

        self._current_task.output_files.append(filename)

        if os.path.exists(filename):
            return filename

        temp_file = filename + '.tmp'

        build_fn(temp_file)

        if crc != self._current_task.complete_crc():
            raise RuntimeError(f'Dependencies of task {self._current_task.id} have changed after output() invocation')

        os.rename(temp_file, filename)
        return filename


    @property
    def is_running(self) -> bool:
        return self._running


_BUILDER = None  # type: Union[Builder, None]


def task(f):
    t = Task(f.__name__, f)

    @wraps(f)
    def task_fn():
        global _BUILDER
        if _BUILDER:
            return _BUILDER.eval(t.id)
        else:
            return f()

    task_fn._builder_task = t

    return task_fn


def reg_src(f: str) -> str:
    global _BUILDER
    if _BUILDER:
        _BUILDER.reg_src(f)
    return f


def output(name: str, build_fn: Callable[[str], None]) -> str:
    global _BUILDER
    if not _BUILDER:
        raise RuntimeError('No builder')
    return _BUILDER.output(name, build_fn)


def temp(filename: str) -> str:
    global _BUILDER
    temp_dir = _BUILDER.temp_dir if _BUILDER else 'tmp'
    return os.path.join(temp_dir, filename)


def builder_session():
    global _BUILDER
    if not _BUILDER:
        raise RuntimeError('No builder')
    return _BUILDER.session()


def set_constant(name: str, val):
    global _BUILDER
    if not _BUILDER:
        raise RuntimeError('No builder')
    if _BUILDER.is_running:
        raise RuntimeError('set_constant() is not allowed within running builder')
    _BUILDER.set_constant(name, val)


def set_builder(modules, temp_dir='tmp'):
    global _BUILDER
    if _BUILDER and _BUILDER.is_running:
        raise RuntimeError('set_builder() is not allowed within running builder')
    _BUILDER = Builder(modules, temp_dir=temp_dir)


__all__ = [
    'Builder',
    'task',
    'reg_src',
    'output',
    'temp',
    'builder_session',
    'set_constant',
    'set_builder'
]
