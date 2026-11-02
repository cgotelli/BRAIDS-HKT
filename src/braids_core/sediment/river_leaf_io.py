"""Authenticated immutable prepared meshes and explicit mesh-epoch output."""
from pathlib import Path
import hashlib
import json
import re
import time
import tempfile
from types import MappingProxyType
import numpy as np
from braids_gpu.atomic_output import save_arrays_atomic,write_json_atomic
from braids_gpu.atomic_output import WINDOWS_TRANSIENT_REPLACE_ERRORS
from braids_core.windows_paths import filesystem_path


def sha(path):return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def publish_directory(source, target, cancelled=lambda:None):
    """Publish a closed private directory, never replace an existing cache.

    Windows can retain a short-lived reader/scanner lock after synchronous
    export. Retry only its access/sharing errors, with a bounded delay and Stop
    checks. Persistent errors remain failures and private evidence is retained.
    """
    source, target = filesystem_path(source), filesystem_path(target)
    delay = .02
    for attempt in range(8):
        cancelled()
        if target.exists() or target.is_symlink():
            raise FileExistsError('Prepared cache target already exists: '+str(target))
        try:
            source.rename(target)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in WINDOWS_TRANSIENT_REPLACE_ERRORS or attempt == 7:
                raise
            time.sleep(delay)
            delay = min(1., 2*delay)


class PreparedLeafCache:
    def __init__(self,root,signature,prepare,cancelled=lambda:None):
        self.root=filesystem_path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.signature=signature;self.prepare=prepare;self.records=[];self.cancelled=cancelled
        # Arrays in this pool are backed by immutable bytes, not a caller's
        # editable buffers. Only detached working copies may leave the cache.
        self._payloads={}
        self._receipts={}
        self._documents={}

    @staticmethod
    def _freeze_payload(payload):
        if not isinstance(payload,dict) or set(payload)!={'data','fields'}:
            raise ValueError('Prepared leaf payload requires data and fields')
        groups={}
        for group in ('data','fields'):
            values=payload[group]
            if not isinstance(values,dict) or not values:
                raise ValueError('Prepared leaf payload requires nonempty mappings')
            if any(not isinstance(name,str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) is None for name in values):
                raise ValueError('Invalid prepared leaf array name')
            frozen={}
            for name,value in sorted(values.items()):
                array=np.asarray(value)
                if array.dtype.kind not in 'biufcSU':
                    raise ValueError('Prepared leaf payload requires numeric or string arrays')
                # Even setflags(write=True) cannot make these private arrays
                # writable. Preserve shape, dtype, signed zero and exact bytes.
                frozen[name]=np.frombuffer(array.tobytes(order='C'),dtype=array.dtype).reshape(array.shape)
            groups[group]=MappingProxyType(frozen)
        return MappingProxyType(groups)

    @staticmethod
    def _payload_digest(payload):
        result=hashlib.sha256()
        for group in ('data','fields'):
            for name,value in sorted(payload[group].items()):
                array=np.asarray(value)
                header=json.dumps([group,name,array.dtype.str,list(array.shape)],separators=(',',':')).encode()
                # Private arrays are contiguous, immutable byte-backed views.
                # Hash their bytes directly instead of allocating another full
                # copy. An injected noncontiguous value still uses canonical C
                # ordering and is checked against the same original digest.
                raw=memoryview(np.ascontiguousarray(array).reshape(-1).view(np.uint8))
                result.update(len(header).to_bytes(8,'little'));result.update(header)
                result.update(raw.nbytes.to_bytes(8,'little'));result.update(raw)
        return result.hexdigest()

    def _folder(self,folder):
        folder=filesystem_path(folder)
        root=self.root.resolve()
        resolved=folder.resolve()
        if (folder.is_symlink() or
            re.fullmatch(r'[0-9a-f]{20}',folder.name) is None or
            resolved.parent!=root):
            raise ValueError('Prepared leaf payload must belong to this cache')
        return folder,resolved

    @staticmethod
    def _member(folder,name,*,checked_directories=None,resolved_folder=None):
        path=Path(name)
        if path.is_absolute() or path.drive or '..' in path.parts or not path.parts:
            raise ValueError('Prepared mesh cache member escapes entry')
        target=folder/path
        # One verification inspects a shared ancestor only once. In particular,
        # the many flat payload/sidecar files need not each walk back to C:\.
        # Resolve nested directories once as well, retaining junction/escape
        # rejection without resolving every ordinary file through that chain.
        checked=set() if checked_directories is None else checked_directories
        resolved=folder.resolve() if resolved_folder is None else resolved_folder
        for parent in target.parents:
            if parent==folder.parent or parent in checked:continue
            if parent.is_symlink():
                raise ValueError('Prepared mesh cache linked or missing file: '+name)
            if parent.is_relative_to(folder) and not parent.resolve().is_relative_to(resolved):
                raise ValueError('Prepared mesh cache member escapes entry')
            checked.add(parent)
        if target.is_symlink() or not target.is_file():
            raise ValueError('Prepared mesh cache linked or missing file: '+name)
        return target

    def _verify(self,folder,identity=None):
        self.cancelled();folder,resolved=self._folder(folder)
        directories={folder} # _folder already checked this directory itself.
        def member(name):
            return self._member(folder,name,checked_directories=directories,resolved_folder=resolved)
        manifest=member('CACHE.json');raw=manifest.read_bytes()
        receipt=hashlib.sha256(raw).hexdigest()
        previous=self._receipts.get(folder.name)
        if previous is not None and previous!=receipt:
            raise ValueError('Prepared mesh cache receipt changed')
        cached=self._documents.get(folder.name)
        if cached is not None and cached[0]==receipt:
            saved=cached[1]
        else:
            saved=json.loads(raw)
            if (set(saved) not in ({'identity','files'},{'identity','files','payload_sha256'}) or
                not isinstance(saved['files'],dict) or not isinstance(saved['identity'],dict)):
                raise ValueError('Prepared mesh cache receipt differs')
            if fingerprint(saved['identity'])[:20]!=folder.name:
                raise ValueError('Prepared mesh cache identity differs')
        expected=saved['identity']
        if expected.get('signature')!=self.signature or (identity is not None and expected!=identity):
            raise ValueError('Prepared mesh cache identity differs')
        actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        if actual!=set(saved['files'])|{'CACHE.json'}:
            raise ValueError('Prepared mesh cache file inventory differs')
        for name,digest in saved['files'].items():
            self.cancelled()
            if sha(member(name))!=digest:
                raise ValueError('Prepared mesh cache file differs: '+name)
        if manifest.read_bytes()!=raw:
            raise ValueError('Prepared mesh cache receipt changed during verification')
        self._receipts[folder.name]=receipt
        self._documents[folder.name]=(receipt,saved)
        return saved

    def get(self,leaves,mesh):
        self.cancelled()
        began=time.perf_counter()
        identity=dict(signature=self.signature,leaves=list(leaves),
            polygons=[np.asarray(c.vertices_xy).tolist() for c in mesh.cells])
        # Match the shared-runtime cache's bounded spelling. The full identity
        # and every file hash remain verified: a prefix collision fails closed.
        key=fingerprint(identity);folder=self.root/key[:20];manifest=folder/'CACHE.json';hit=manifest.exists()
        if hit:
            self._verify(folder,identity)
        else:
            if folder.exists():raise FileExistsError('Unpublished legacy cache entry exists')
            # Publish complete geometry only. A cancelled private attempt stays
            # identifiable for diagnosis but cannot poison this cache key.
            pending=filesystem_path(tempfile.mkdtemp(prefix=key[:20]+'.partial-',dir=self.root))
            try:
                payload=self.prepare(pending,leaves,mesh)
                frozen=None
                # The previous file-only callback contract ignored its return
                # value (some simple writers return a byte count). Only a
                # mapping opts into the explicit resident-payload contract.
                if isinstance(payload,dict):
                    frozen=self._freeze_payload(payload)
                    # Serialization belongs to this cache so the resident
                    # values and persisted bytes have one construction owner.
                    for group,name in (('data','records.npz'),('fields','fields.npz')):
                        if (pending/name).exists() or (pending/name).is_symlink():
                            raise ValueError('Prepared callback must not write '+name)
                        self.cancelled();save_arrays_atomic(pending/name,**frozen[group])
                files={p.relative_to(pending).as_posix():sha(p) for p in sorted(pending.rglob('*')) if p.is_file()}
                receipt=dict(identity=identity,files=files)
                if frozen is not None:receipt['payload_sha256']=self._payload_digest(frozen)
                write_json_atomic(pending/'CACHE.json',receipt)
                publish_directory(pending,folder,self.cancelled)
                self._receipts[folder.name]=sha(folder/'CACHE.json')
                if frozen is not None:self._payloads[folder.name]=frozen
            except BaseException as exc:
                write_json_atomic(pending/'ABORTED.json',dict(identity=identity,error=f'{type(exc).__name__}: {exc}'))
                raise
        self.records.append(dict(key=key,hit=hit,elapsed_s=time.perf_counter()-began))
        return folder

    def load_payload(self,folder):
        """Load authenticated leaf inputs; never expose resident cache arrays.

        Every call checks the immutable file inventory, including resident hits.
        Fresh processes decode NPZ exactly once, with checksums before and after
        decoding. The canonical payload digest binds dtype/shape/value and is
        independent of a path or ZIP timestamp.
        """
        folder=filesystem_path(folder);saved=self._verify(folder)
        if ('payload_sha256' not in saved or
            not {'records.npz','fields.npz'}.issubset(saved['files'])):
            raise ValueError('Prepared leaf has no typed resident payload')
        payload=self._payloads.get(folder.name)
        if payload is None:
            values={}
            for group,name in (('data','records.npz'),('fields','fields.npz')):
                self.cancelled()
                with np.load(self._member(folder,name),allow_pickle=False) as archive:
                    if len(archive.files)!=len(set(archive.files)):
                        raise ValueError('Prepared leaf archive has duplicate arrays')
                    values[group]={k:archive[k] for k in archive.files}
            payload=self._freeze_payload(values)
            if self._verify(folder)!=saved:
                raise ValueError('Prepared leaf changed while loading payload')
        if self._payload_digest(payload)!=saved['payload_sha256']:
            raise ValueError('Prepared leaf payload digest differs')
        self._payloads[folder.name]=payload
        self.cancelled()
        def detached(group):
            return {name:(value.item() if value.ndim==0 else value.copy())
                    for name,value in payload[group].items()}
        return detached('data'),detached('fields')


class MeshEpochWriter:
    def __init__(self,root):
        self.root=filesystem_path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.index=self.root/'INDEX.json';self.frames=[];self.epochs={}
        if self.index.exists():
            doc=json.loads(self.index.read_text());self.frames=doc['frames'];self.epochs=doc['epochs']
            for name,digest in {**self.epochs,**{r['file']:r['sha256'] for r in self.frames}}.items():
                if sha(self.root/name)!=digest:raise ValueError('Mesh epoch output checksum differs')

    def write(self,controller,fields):
        leaves=controller.leaves
        key=fingerprint(dict(geometry=controller.geometry_sha256,leaves=list(leaves)))
        name='mesh_'+key+'.npz';path=self.root/name
        if name not in self.epochs:
            if path.exists():raise FileExistsError('Unindexed mesh epoch exists')
            # A published epoch already owns immutable geometry. Only a new
            # epoch needs it, and the controller normally holds that mesh.
            # Physical fields below are always written from the current owner.
            mesh=controller._mesh_cache.get(tuple(leaves))
            if mesh is None:mesh=controller.hierarchy.select(leaves)
            offsets=np.r_[0,np.cumsum([len(c.vertices_xy) for c in mesh.cells])]
            save_arrays_atomic(path,leaves=np.asarray(leaves),levels=controller.hierarchy.node_level[list(leaves)],
                offsets=offsets,vertices=np.concatenate([c.vertices_xy for c in mesh.cells]),area=np.array([c.area_m2 for c in mesh.cells]))
            self.epochs[name]=sha(path)
        frame=f'frame_{controller.owner.steps:08d}.npz'
        if (self.root/frame).exists():raise FileExistsError('Accepted frame already exists')
        if any(np.asarray(v).shape[-1:]!=(len(leaves),) for k,v in fields.items() if k in ('q','z','mass')):raise ValueError('Frame shape differs from mesh epoch')
        save_arrays_atomic(self.root/frame,mesh_epoch=np.array(name),**fields)
        self.frames.append(dict(file=frame,sha256=sha(self.root/frame),epoch=name,time_s=controller.owner.time,steps=controller.owner.steps))
        write_json_atomic(self.index,dict(epochs=self.epochs,frames=self.frames))
