from contextlib import contextmanager
from functools import wraps, reduce
from typing import Callable, Any, Mapping, TypeVar, NamedTuple, Union, MutableMapping, MutableSet, Set, List
from datetime import datetime
from zlib import crc32
import os


T = TypeVar("T")
Id = str
Mtime = float


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


class Rec:
    id: Id
    f: Any
    value: Any
    deps: MutableMapping[Id, Any]
    src_files: MutableSet[str]
    out_files: MutableSet[str]
    mtime: Mtime
    epoch: int

    def __init__(self, task):
        self.id = task.id
        self.f = task.f
        self.value = None
        self.deps = {}
        self.src_files = set()
        self.out_files = set()
        self.mtime = 0.0
        self.epoch = 0

    def reset(self):
        self.deps.clear()
        self.src_files.clear()
        self.out_files.clear()
        self.mtime = 0


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


    def _eval(self, task_id: Id):
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
            self._current_task.deps[rec.id] = rec.value

        return rec.value


    def _compute(self, rec: Rec):
        if rec.epoch == self._epoch:
            return

        if rec.epoch == 0 or self._has_changed_deps(rec):
            rec.reset()
            with self._new_task_ctx(rec):
                rec.value = rec.f()
                if rec.src_files:
                    rec.mtime = datetime.now().timestamp()

        rec.epoch = self._epoch


    def _has_changed_deps(self, rec: Rec):
        with self._new_task_ctx(None):
            for dep, old_val in rec.deps.items():
                if old_val is not self._eval(dep):
                    return True

        src_mtime = max((get_mtime(f) for f in rec.src_files), default=0)

        for out in rec.out_files:
            if src_mtime > get_mtime(out):
                return True

        return src_mtime > rec.mtime


    def reg_src(self, f: str):
        if self._current_task:
            self._current_task.src_files.add(os.path.abspath(f))


    def reg_out(self, f: str):
        if self._current_task:
            self._current_task.out_files.add(os.path.abspath(f))


    def current_src_files(self) -> Union[Set[str], None]:
        return self._current_task and self._current_task.src_files


    def set_constant(self, name: str, val):
        if self._current_task:
            raise RuntimeError('set_constant() is not allowed within a task')
        const = lambda: val
        const.__name__ = name
        self._module[name] = task(const)


    @contextmanager
    def _session(self):
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
            return self._eval(task_id)
        else:
            with self._session():
                return self._eval(task_id)


    def run(self, f: Callable[[], T]) -> T:
        try:
            t = task._builder_task
        except AttributeError:
            t = None

        if self._running:
            return self._eval(t.id) if t else f()
        else:
            with self._session():
                return self._eval(t.id) if t else f()


    @property
    def has_task(self) -> bool:
        return self._current_task is not None


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


def filename(name: str) -> str:
    global _BUILDER
    temp_dir = _BUILDER.temp_dir if _BUILDER else os.path.abspath('tmp')
    file = os.path.join(temp_dir, name)
    return os.path.normpath(file)


def output(f: str) -> str:
    global _BUILDER
    if _BUILDER and _BUILDER.current_src_files():
        name, ext = os.path.splitext(f)
        checksum = reduce(lambda crc, src: crc32(src.encode(), crc), sorted(_BUILDER.current_src_files()), 0)
        f = os.path.basename(name) + '-' + '{:x}'.format(checksum) + ext
    return filename(f)


def reg_src(f: str) -> str:
    global _BUILDER
    if _BUILDER:
        _BUILDER.reg_src(f)
    return f


def is_fresh(f: str, *deps) -> bool:
    global _BUILDER

    if _BUILDER:
        _BUILDER.reg_out(f)

    try:
        mtime = os.path.getmtime(f)
    except FileNotFoundError:
        return False

    if not deps and _BUILDER and _BUILDER.current_src_files():
        deps = _BUILDER.current_src_files()

    for dep in deps:
        try:
            dep_mtime = os.path.getmtime(dep)
            if mtime < dep_mtime:
                return False
        except FileNotFoundError:
            pass

    return True


def set_default_builder(modules, temp_dir='tmp'):
    global _BUILDER
    if _BUILDER and _BUILDER.has_task:
        raise RuntimeError('set_default_builder() is not allowed within a task')

    _BUILDER = Builder(modules, temp_dir=temp_dir)


def set_constant(name: str, val):
    global _BUILDER
    if not _BUILDER:
        raise RuntimeError('There is no default builder set')

    _BUILDER.set_constant(name, val)


def compile(src_files_glob: str, out_ext: str, transform: Callable[[List[str], str], None]) -> str:
    import glob

    src_files = glob.glob(src_files_glob)
    if not src_files:
        raise RuntimeError(f'No files matching {src_files_glob}')

    for src in src_files:
        reg_src(src)

    if len(src_files) == 1:
        out_name = src_files[0]
    else:
        out_name = os.path.dirname(src_files[0])

    out_name, _ = os.path.splitext(out_name)
    out_file = output(out_name + out_ext)

    if not is_fresh(out_file):
        transform(src_files, out_file)

    return out_file


__all__ = [
    'Builder',
    'task',
    'filename',
    'output',
    'is_fresh',
    'set_default_builder',
    'set_constant',
    'compile'
]
