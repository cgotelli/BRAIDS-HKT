"""In-process, zero-step records from the frozen native geometry initializer.

This is an I/O optimization, not another geometry implementation. The locked
BRAIDS exporter module is compiled unchanged except for its record writers:
those synchronously copy records through a C callback instead of writing .bin
files. Calls share a private, serialized DLL but release all its mesh arrays
after copying each result. The case.data carrier remains an input, and no
hydraulic step is executed here.

Only the private, validated inert carrier written by adaptive_river_worker is
admitted. This is deliberately not a general public native-deck reader.
"""
from __future__ import annotations

import ctypes
import atexit
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

from braids_core.windows_native_io import native_directory
from braids_core.windows_paths import filesystem_path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'braids_gpu/native_exporter_v1'
METHOD = 'frozen_geometry_in_process_records_v132'
_INITIALIZER_LIMIT = 2
_INITIALIZERS = OrderedDict()
_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p)

_DECLARATIONS = '''
 abstract interface
  subroutine river_record_callback(unit,name,kind,rank,dims,values) bind(C)
   use iso_c_binding
   integer(c_int),value :: unit,kind,rank
   character(c_char),intent(in) :: name(64)
   integer(c_int),intent(in) :: dims(4)
   type(c_ptr),value :: values
  end subroutine
 end interface
 procedure(river_record_callback),pointer :: river_record_sink=>null()
'''

_WRITERS = '''
 subroutine gpu_put_header(unit,name,kind,dimensions)
  integer,intent(in) :: unit,kind,dimensions(:)
  character(*),intent(in) :: name
  ! All supported writers dispatch the complete header and value together.
  error stop 'standalone record header is not supported by memory exporter'
 end subroutine
 subroutine gpu_put_r(unit,name,values,dimensions)
  use iso_c_binding
  integer,intent(in) :: unit,dimensions(:)
  character(*),intent(in) :: name
  real(8),target,intent(in) :: values(:)
  character(64),target :: padded
  integer(c_int),target :: dims(4)
  type(c_ptr) :: pointer
  padded=name;dims=0;dims(1:size(dimensions))=dimensions
  pointer=c_null_ptr
  if(size(values)>0) pointer=c_loc(values(1))
  call river_record_sink(int(unit,c_int),padded,2_c_int,int(size(dimensions),c_int),dims,pointer)
 end subroutine
 subroutine gpu_put_i(unit,name,values,dimensions)
  use iso_c_binding
  integer,intent(in) :: unit,dimensions(:)
  character(*),intent(in) :: name
  integer,target,intent(in) :: values(:)
  character(64),target :: padded
  integer(c_int),target :: dims(4)
  type(c_ptr) :: pointer
  padded=name;dims=0;dims(1:size(dimensions))=dimensions
  pointer=c_null_ptr
  if(size(values)>0) pointer=c_loc(values(1))
  call river_record_sink(int(unit,c_int),padded,1_c_int,int(size(dimensions),c_int),dims,pointer)
 end subroutine
 subroutine gpu_finish(unit)
  use iso_c_binding
  integer,intent(in) :: unit
  character(64) :: padded
  integer(c_int) :: dims(4)
  padded='END';dims=0
  call river_record_sink(int(unit,c_int),padded,0_c_int,0_c_int,dims,c_null_ptr)
 end subroutine
 subroutine gpu_open(unit,path)
  integer,intent(out) :: unit
  character(*),intent(in) :: path
  select case(path)
  case('coupled_input.bin')
   unit=1
  case('normal_metadata.bin')
   unit=2
  case default
   error stop 'unexpected record file outside inert memory exporter'
  end select
 end subroutine

 subroutine river_records_load(path,npath,nthreads,sink,code) bind(C)
  use iso_c_binding
  use omp_lib,only:omp_set_num_threads,omp_set_dynamic,omp_get_max_threads,omp_get_dynamic
  character(c_char),intent(in) :: path(*)
  integer(c_int),value :: npath,nthreads
  type(c_funptr),value :: sink
  integer(c_int),intent(out) :: code
  character(len=:),allocatable :: filename
  integer :: i,old_threads
  logical :: old_dynamic
  code=1
  old_threads=omp_get_max_threads();old_dynamic=omp_get_dynamic()
  call omp_set_dynamic(.false.);call omp_set_num_threads(nthreads)
  allocate(character(len=npath)::filename)
  do i=1,npath
   filename(i:i)=path(i)
  enddo
  call load_case(filename)
  if(nstate/=5.or.nfrac/=1.or.nbed_layers/=1.or.amr_enabled.or.scheme/=6) goto 900
  if(bedlaw/=0.or.exchange_mode/=0.or.secondary_flow.or.horizontal_viscosity/=0) goto 900
  if(general_avalanche.or.bank_failure.or.bedload_adaptation.or.bedform_roughness) goto 900
  if(any(q(4,:)/=0).or.any(fraction_concentration/=0).or.any(fraction_series_concentration/=0)) goto 900
  ! Exact state of the original executable under export_environment's legacy
  ! zero-step controls. These fields describe the inert input carrier only.
  suspended_transport_active=.false.;cpu_time_integrator=33;cpu_outlet_extension=0
  call c_f_procpointer(sink,river_record_sink)
  call gpu_export_input()
  call normal_export_metadata()
  nullify(river_record_sink)
  code=0
900 continue
  call omp_set_num_threads(old_threads);call omp_set_dynamic(old_dynamic)
 end subroutine
'''


def _sha(path):
    with filesystem_path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _release_source(original):
    """Generate explicit cleanup for every module-owned allocatable array."""
    declarations = original.split('\ncontains\n', 1)[0]
    declarations = '\n'.join(row.split('!', 1)[0] for row in declarations.splitlines())
    declarations = re.sub(r'&\s*\n\s*&?', '', declarations)
    names = []
    for row in declarations.splitlines():
        if re.search(r'\ballocatable\b', row, re.I) is None:
            continue
        if '::' not in row:
            raise ValueError('Unrecognized frozen module allocation declaration')
        rhs = row.split('::', 1)[1]
        found = re.findall(r'([A-Za-z]\w*)\s*\([^)]*\)', rhs)
        remainder = re.sub(r'[A-Za-z]\w*\s*\([^)]*\)', '', rhs)
        if not found or remainder.replace(',', '').strip():
            raise ValueError('Unrecognized frozen allocatable array list')
        names.extend(found)
    if len(names) != len(set(names)) or len(names) < 100:
        raise ValueError('Frozen allocation inventory differs')
    body = ['subroutine river_records_release() bind(C)',
            ' nullify(river_record_sink)']
    body.extend(f' if(allocated({name})) deallocate({name})' for name in names)
    body.append('end subroutine')
    return '\n'.join(body) + '\n', names


def build():
    """Build-time binding; portable releases replace this with a locked DLL."""
    compiler = shutil.which('gfortran')
    if not compiler:
        raise RuntimeError('Development memory-exporter build requires gfortran')
    receipt = json.loads((SOURCE / 'build.json').read_text())
    parts = [SOURCE / Path(value).name for value in receipt['command']
             if value.lower().endswith(('.f', '.f90'))]
    if not parts or parts[-1].name != 'cutcell_solver.f90':
        raise ValueError('Frozen geometry exporter source order differs')
    hashes = {path.name: _sha(path) for path in parts}
    original = parts[-1].read_text()
    begin = original.index(' subroutine gpu_put_header(')
    end = original.index(' subroutine gpu_export_input()', begin)
    release, allocations = _release_source(original)
    code = original[:begin] + _WRITERS + '\n' + release + original[end:]
    code = code[:code.index('\nend module') + len('\nend module')] + '\n'
    marker = '\ncontains\n'
    if code.count(marker) != 1:
        raise ValueError('Frozen geometry exporter module layout differs')
    code = code.replace(marker, '\n' + _DECLARATIONS + marker)
    version = subprocess.check_output([compiler, '--version'], text=True).splitlines()[0]
    material = dict(sources=hashes, generated=hashlib.sha256(code.encode()).hexdigest(),
                    compiler=version, method=METHOD, released_allocations=allocations)
    key = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:20]
    folder = ROOT / 'build/river_mesh_records' / key
    folder.mkdir(parents=True, exist_ok=True)
    library = folder / ('mesh_records.dll' if os.name == 'nt' else 'mesh_records.so')
    if not library.exists():
        # Generated build artifact; the archived input sources are never edited.
        generated = folder / 'mesh_records.f90'
        generated.write_text(code)
        command = [compiler, '-O3', '-flto', '-fopenmp', '-shared',
                   '-ffree-line-length-none', '-ffixed-line-length-none',
                   *map(str, parts[:-1]), str(generated), '-o', str(library)]
        if os.name == 'nt':
            command += ['-Wl,--export-all-symbols']
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=180)
        (folder / 'build.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('Memory-exporter compilation failed: ' + str(folder / 'build.log'))
        if hashes != {path.name: _sha(path) for path in parts}:
            raise RuntimeError('Frozen geometry sources changed during compilation')
        identity = dict(material, library_sha256=_sha(library), command=command,
                        source_geometry_unchanged=True, hydraulic_steps=0)
        (folder / 'identity.json').write_text(json.dumps(identity, indent=2))
    identity = json.loads((folder / 'identity.json').read_text())
    if identity['sources'] != hashes or _sha(library) != identity['library_sha256']:
        raise ValueError('Memory-exporter source/library identity differs')
    return library, identity, compiler


def _merged(records):
    """Match the frozen snapshot reader's explicit, non-physical aliases."""
    if set(records) != {1, 2}:
        raise ValueError('Memory exporter record groups differ')
    data = dict(records[1])
    if (data.get('nstate') != 5 or data.get('nfrac') != 1 or
            data.get('nbed_layers') != 1 or data.get('bedlaw') != 0 or
            data.get('exchange_mode') != 0 or data.get('horizontal_viscosity') != 0):
        raise ValueError('Memory exporter requires the inert hydraulic carrier')
    for key, value in records[2].items():
        if key in data and not np.array_equal(data[key], value):
            raise ValueError('Memory exporter duplicate record differs: ' + key)
        data[key] = value
    data['gaia_fixed_inlet_contract'] = 'historical_absent'
    data['q'] = np.ascontiguousarray(data['state'][:3])
    data['dry'] = data['dry_tolerance']
    data['outlet_extension'] = data['cpu_outlet_extension']
    data['dt'] = data['dtmax']
    data['bedoffset'] = data['boundary_bed_offset']
    data['bc_hydraulic'] = np.ascontiguousarray(data['forcing'][:2])
    data['bc_bed_input'] = data['fraction_bed_input'].copy()
    for key in ('section_seed_node', 'boundary_seed_position'):
        value = np.asarray(data['native_' + key])
        if value.dtype != np.int32 or value.shape != (data['nb'],) or np.any(value < 0):
            raise ValueError('Memory exporter invalid boundary seed: ' + key)
        data[key] = np.ascontiguousarray(value - 1)
    return data


def _validate_carrier(deck):
    """Authenticate the current writer's compact v24/v10 inert input only.

    This guards accidental corruption before Fortran's ERROR STOP parser. It
    does not admit arbitrary user-written legacy decks or reconstructed hashes.
    Public input validation remains the common project/worker resolver.
    """
    metadata_path = deck.with_suffix('.data.json')
    binary = deck.with_suffix('.data.bin')
    for path in (metadata_path, binary):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Memory exporter requires regular generated carrier sidecars')
    metadata = json.loads(metadata_path.read_text())
    if (metadata.get('format') != 'xbraids_cutcell-v24' or
            metadata.get('state_layout_format') != 'xbraids_cutcell-v10' or
            metadata.get('amr') is not None or
            metadata.get('sidecars', {}).get('native_binary') != binary.name):
        raise ValueError('Memory exporter accepts only the current inert v24/v10 carrier')
    raw = deck.read_bytes()
    if (not raw.endswith(b'\n') or
            any(value < 32 and value != 10 for value in raw) or 127 in raw):
        raise ValueError('Memory exporter carrier byte layout differs')
    lines = raw.decode('ascii').removesuffix('\n').split('\n')
    if len(lines) < 9:
        raise ValueError('Truncated memory-exporter carrier')
    expected = lines.pop(1)
    binary_hash = _sha(binary)
    if (lines[0] != '24' or lines[2] != '10' or
            lines[1] != binary.name + ' ' + binary_hash or
            metadata.get('native_binary_sha256') != binary_hash or
            metadata.get('fingerprint') != expected):
        raise ValueError('Memory exporter carrier/binary identity differs')
    dimensions = list(map(int, lines[3].split()))
    if (len(dimensions) != 5 or dimensions[2] < 0 or
            min(dimensions[0], dimensions[1], dimensions[3], dimensions[4]) < 1):
        raise ValueError('Memory exporter invalid carrier dimensions')
    nc, nf, nb, nx, ny = dimensions
    parameters = np.asarray(list(map(float, lines[4].split())))
    times = list(map(float, lines[5].split()))
    if (len(parameters) != 13 or len(times) != 3 or
            not np.isfinite(parameters).all() or not np.isfinite(times).all() or
            min(times) <= 0 or lines[6] != '6 0 0'):
        raise ValueError('Memory exporter requires finite inert carrier controls')
    normalized = list(lines)
    normalized[5] = 'END ' + ' '.join(lines[5].split()[1:])
    calculated = hashlib.sha256(('\n'.join(normalized) + json.dumps(
        metadata['source_sha256'], sort_keys=True)).encode()).hexdigest()
    if calculated != expected:
        raise ValueError('Memory exporter carrier fingerprint differs')
    # Binary dimensions are checked before any Fortran allocation/indexing.
    with binary.open('rb') as stream:
        if stream.read(16) != b'BRAIDS_NATIVE_V1':
            raise ValueError('Memory exporter binary magic differs')
        header = np.fromfile(stream, dtype='<i4', count=9)
        if (header.shape != (9,) or not np.array_equal(header,
                [1, nc, nf, 5, 0, 0, 10, 0, 0]) or
                binary.stat().st_size != 52 + 92*nc + 80*nf):
            raise ValueError('Memory exporter binary schema/size differs')
    return {path.name: _sha(path) for path in (deck, metadata_path, binary)}


class _ResidentInitializer:
    """One authenticated loader, with no live mesh between calls.

    There is no __del__: explicit cache eviction/shutdown runs while ctypes,
    filesystem helpers and the native serialization lock are still available.
    """
    def __init__(self, library, identity, compiler):
        from .river_runtime_lease import RuntimeFileLease
        self.expected = identity['library_sha256']
        self.lease = RuntimeFileLease()
        self.directory = None
        self.lib = self.load = self.release = None
        self.closed = False
        directories = []
        try:
            if self.lease.digest(library, immutable=True) != self.expected:
                raise ValueError('Memory exporter library differs before loading')
            # No TemporaryDirectory weakref finalizer: it may run before our
            # atexit loader cleanup and attempt to delete a still-loaded DLL.
            self.directory = Path(tempfile.mkdtemp(prefix='braids-records-'))
            self.path = self.directory / library.name
            shutil.copy2(library, self.path)
            if self.lease.digest(self.path, immutable=True) != self.expected:
                raise ValueError('Memory exporter private library copy differs')
            if os.name == 'nt':
                for folder in dict.fromkeys((str(library.parent), str(Path(compiler).parent))):
                    directories.append(os.add_dll_directory(folder))
            self.lib = ctypes.CDLL(str(self.path))
            self.release = self.lib.river_records_release
            self.release.argtypes = []
            self.release.restype = None
            self.load = self.lib.river_records_load
            self.load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                                  _CALLBACK_TYPE, ctypes.POINTER(ctypes.c_int)]
            self.load.restype = None
        except BaseException:
            self.close()
            raise
        finally:
            # Dependencies are eagerly resolved; retain neither search cookies
            # nor a changed process PATH between mesh initializations.
            for directory in directories:
                directory.close()

    def verify(self, library):
        if self.closed:
            raise RuntimeError('Memory exporter loader is closed')
        if (self.lease.digest(library, immutable=True) != self.expected or
                self.lease.digest(self.path, immutable=True) != self.expected):
            raise ValueError('Memory exporter leased library identity differs')

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.lib is not None:
                if self.release is not None:
                    self.release()
                import _ctypes
                if os.name == 'nt':
                    _ctypes.FreeLibrary(self.lib._handle)
                else:
                    _ctypes.dlclose(self.lib._handle)
                self.lib = self.load = self.release = None
        finally:
            self.lease.close()
            if self.directory is not None:
                # This directory contains only our private DLL. Remove exactly
                # that file, then the empty directory; never recurse on a path.
                self.path.unlink(missing_ok=True)
                self.directory.rmdir()
                self.directory = None


def _resident(library, identity, compiler):
    """Called only under bend_native_hydraulics._lock."""
    key = identity['library_sha256']
    value = _INITIALIZERS.get(key)
    if value is None:
        while len(_INITIALIZERS) >= _INITIALIZER_LIMIT:
            _, old = _INITIALIZERS.popitem(last=False)
            old.close()
        value = _ResidentInitializer(library, identity, compiler)
        _INITIALIZERS[key] = value
    value.verify(library)
    _INITIALIZERS.move_to_end(key)
    return value


def close_initializers():
    """Release process-owned private loaders; safe to call more than once."""
    from .bend_native_hydraulics import _lock
    with _lock:
        while _INITIALIZERS:
            _, value = _INITIALIZERS.popitem(last=False)
            value.close()


atexit.register(close_initializers)


def initialize(deck, *, threads=8, cancelled=lambda: None):
    """Read a private validated carrier and return (owned records, identity).

    Stop is checked before/after native initialization and at every record copy.
    Callback exceptions are captured and reraised only after native code returns.
    All ndarray storage is copied before every native mesh allocation is freed.
    """
    if type(threads) is not int or not 1 <= threads <= 256:
        raise ValueError('Memory exporter requires 1..256 threads')
    cancelled()
    deck = filesystem_path(deck)
    if deck.name != 'case.data' or deck.is_symlink() or not deck.is_file():
        raise ValueError('Memory exporter requires a private regular case.data carrier')
    deck = deck.resolve()
    from .river_runtime_lease import RuntimeFileLease
    lease = RuntimeFileLease()
    try:
        # These inputs are immutable for this one initialization, not cached
        # between edits. Deny Windows writes/deletion before validation starts.
        for path in (deck, deck.with_suffix('.data.json'), deck.with_suffix('.data.bin')):
            if path.is_symlink() or not path.is_file():
                raise ValueError('Memory exporter requires regular generated sidecars')
            lease.digest(path, immutable=True)
        return _initialize(deck, threads=threads, cancelled=cancelled, lease=lease)
    finally:
        lease.close()


def _initialize(deck, *, threads, cancelled, lease):
    from .bend_native_hydraulics import _lock
    from .river_runtime_lease import RuntimeFileLease
    # A validated writer's receipt pins the complete deck and source identity.
    # Native load_case uses ERROR STOP for malformed decks; reject any modified
    # artifact against its own generated receipt before entering that routine.
    inputs = _validate_carrier(deck)
    library, identity, compiler = build()
    cancelled()
    groups = {1: {}, 2: {}}
    finished = set()
    errors = []

    def receive(unit, name_pointer, kind, rank, dims_pointer, values_pointer):
        if errors:
            return
        try:
            cancelled()
            name = ctypes.string_at(name_pointer, 64).decode('ascii').strip()
            if unit not in groups or unit in finished:
                raise ValueError('Memory exporter record group/order differs')
            dims = np.ctypeslib.as_array(ctypes.cast(dims_pointer, ctypes.POINTER(ctypes.c_int32)), shape=(4,))
            if name == 'END':
                if kind != 0 or rank != 0:
                    raise ValueError('Memory exporter terminator differs')
                finished.add(unit)
                return
            if name in groups[unit] or kind not in (1, 2) or not 0 <= rank <= 4:
                raise ValueError('Memory exporter record header differs')
            shape = tuple(map(int, dims[:rank]))
            if any(v < 0 for v in shape):
                raise ValueError('Memory exporter invalid dimensions')
            count = int(np.prod(shape)) if rank else 1
            if count > 100000000:
                raise ValueError('Memory exporter implausible dimensions')
            dtype = np.dtype('int32' if kind == 1 else 'float64')
            if count:
                if not values_pointer:
                    raise ValueError('Memory exporter missing record values')
                ctype = ctypes.c_int32 if kind == 1 else ctypes.c_double
                view = np.ctypeslib.as_array(ctypes.cast(values_pointer, ctypes.POINTER(ctype)), shape=(count,))
                if not np.isfinite(view).all():
                    raise ValueError('Memory exporter nonfinite record: ' + name)
                value = view[0].item() if not rank else np.array(view.reshape(shape, order='F'), order='C', copy=True)
            else:
                value = np.empty(shape, dtype=dtype)
            groups[unit][name] = value
        except BaseException as exc:
            errors.append(exc)

    callback = _CALLBACK_TYPE(receive)
    # The initializer is distinct from every CPU owner's DLL. Reuse the loader
    # only: every case is loaded anew and all mesh arrays are freed in finally.
    with _lock:
        runtime = _resident(library, identity, compiler)
        try:
            with native_directory(deck.parent, inputs, cancelled=cancelled) as folder:
                staged_lease = RuntimeFileLease()
                previous = Path.cwd()
                try:
                    for name, expected in inputs.items():
                        if (lease.digest(deck.parent/name, immutable=True) != expected or
                                staged_lease.digest(folder/name, immutable=True) != expected):
                            raise ValueError('Memory exporter input changed before native initialization')
                    os.chdir(folder)
                    code = ctypes.c_int()
                    runtime.load(b'case.data', 9, threads, callback, ctypes.byref(code))
                finally:
                    os.chdir(previous)
                    staged_lease.close()
            if errors:
                raise errors[0]
            if code.value or finished != {1, 2}:
                raise ValueError('Memory exporter carrier rejected or incomplete records')
        finally:
            runtime.release()
    cancelled()
    return _merged(groups), dict(method=METHOD, native=identity, inputs=inputs,
                                threads=threads, hydraulic_steps=0, binary_records_written=False)
