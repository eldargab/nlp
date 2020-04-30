import os
import struct
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any, NamedTuple, Union, MutableMapping, MutableSet, List, Iterable, Tuple
from zlib import crc32


Id = str
Mtime = float
CRC = int


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


def crc_fn(f, crc: CRC) -> CRC:
    return crc32(f.__code__.co_code, crc)


class Task(NamedTuple):
    id: Id
    f: Callable[[], Any]
    f_crc: CRC


class Rec:
    id: Id
    f: Callable[['App'], Any]
    f_crc: CRC
    value: Any
    deps: MutableMapping[Id, Tuple[Any, CRC]]
    src_files: MutableSet[str]
    output_files: List[str]
    crc: CRC
    epoch: int

    def __init__(self, task: Task):
        self.id = task.id
        self.f = task.f
        self.f_crc = task.f_crc
        self.value = None
        self.deps = OrderedDict()
        self.src_files = set()
        self.output_files = []
        self.crc = self.f_crc
        self.epoch = 0

    def reset(self):
        self.deps.clear()
        self.src_files.clear()
        self.output_files.clear()
        self.crc = self.f_crc

    def reg_dep(self, rec: 'Rec'):
        self.deps[rec.id] = rec.value, rec.crc
        self.crc = crc_int(rec.crc, self.crc)

    def compute(self, app: 'App'):
        self.value = self.f(app)
        self.crc = self.complete_crc()

    def complete_crc(self) -> CRC:
        return crc_files(self.src_files, self.crc)


def task(f):
    t = Task(
        id=f.__name__,
        f=f,
        f_crc=crc_fn(f, 0)
    )

    @wraps(f)
    def task_fn(self: App):
        return self._eval(t).value

    task_fn._app_task = t

    return task_fn


class App:
    temp_dir: str
    _state: MutableMapping[Id, Rec]
    _current_task: Union[Rec, None]
    _epoch: int
    _running: bool


    def __init__(self):
        self.temp_dir = 'tmp'
        self._state = {}
        self._epoch = 1
        self._current_task = None
        self._running = False


    @contextmanager
    def _new_task_ctx(self, rec: Union[Rec, None]):
        prev = self._current_task
        self._current_task = rec
        try:
            yield rec
        finally:
            self._current_task = prev


    @contextmanager
    def session(self):
        if self._running:
            yield
            return
        self._epoch += 1
        self._running = True
        try:
            yield
        finally:
            self._running = False


    def _eval(self, t: Task) -> Rec:
        if not self._running:
            with self.session():
                return self._eval(t)

        rec = self._state.get(t.id)

        if not rec or rec.f_crc != t.f_crc:
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
                rec.compute(self)

        rec.epoch = self._epoch


    def _has_changed_deps(self, rec: Rec):
        for f in rec.output_files:
            if not os.path.exists(f):
                return True

        crc = 0
        with self._new_task_ctx(None):
            for dep_id, (old_val, old_crc) in rec.deps.items():
                dep = self._eval_id(dep_id)
                if not dep:
                    return True
                if dep.value is not old_val:
                    return True
                if dep.crc != old_crc:
                    return True
                crc = crc_int(dep.crc, crc)

        crc = crc_files(rec.src_files, crc)

        return crc != rec.crc


    def _eval_id(self, task_id: Id) -> Union[Rec, None]:
        method = getattr(self, task_id, None)
        if method is None:
            return None
        t = getattr(method, '_app_task', None)
        if t is None:
            return None
        return self._eval(t)


    def reg_src(self, f: str):
        if self._current_task:
            self._current_task.src_files.add(os.path.abspath(f))


    def reg_src_module(self, module):
        file = getattr(module, '__file__', None)
        if file:
            self.reg_src(file)


    def output(self, name: str, build_fn: Callable[[str], None]) -> str:
        if not self._current_task:
            raise RuntimeError('No current task')

        crc = self._current_task.complete_crc()
        path = os.path.abspath(os.path.join(self.temp_dir, name))

        prefix, ext = os.path.splitext(path)
        filename = prefix + '-' + '{:x}'.format(crc) + ext

        self._current_task.output_files.append(filename)

        if os.path.exists(filename):
            return filename

        temp_file = filename + '.tmp'

        build_fn(temp_file)

        if crc != self._current_task.complete_crc():
            raise RuntimeError(f'Dependencies of task {self._current_task.id} have changed after output() invocation')

        os.rename(temp_file, filename)
        return filename


    def reset(self):
        if self._running:
            raise RuntimeError("Builder can't be reset while running")
        else:
            self._state.clear()
