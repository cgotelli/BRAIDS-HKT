"""Package-only loaders for the qualified river arithmetic; no runtime compiler.

The release builder records constructor library bindings from a real CPU run.
The installed loader restores only those bindings, not a state or a geometry.
Every payload byte is verified by the launcher before this module is activated.
Development constructors and historical checkpoints are unchanged.
"""
import ctypes
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import numpy as np

RECORDS={}
ACTIVE=False
ENGINE_CLASSES={
 'braids_core.sediment.river_turbine_flow':'Engine',
 'braids_core.sediment.river_hydrokinetic_stress':'FullStressEngine',
 'braids_core.sediment.bend_methods_v1':'NativeEngine',
 'braids_core.sediment.flow2dh_fixed_bed':'FlowEngine',
 'braids_core.sediment.flow2dh_eh_operator':'ClassicFlowEH',
 'braids_core.sediment.flow2dh_mobile_bed':'MobileEngine',
 'braids_core.sediment.flow2dh_reconstructed_bed':'ReconstructionEngine',
 'braids_core.sediment.flow2dh_geometry_bed':'GeometryEngine',
 'braids_core.sediment.flow2dh_effective_bed':'EffectiveShearEngine',
 'braids_core.sediment.flow2dh_objective_bed':'ObjectiveEngine',
 'braids_core.sediment.shoreline_viscosity':'ViscosityEngine',
 'validation.sediment.river_morphodynamics.sediment_wave':'WaveEngine',
}

def sha(path):
    from .river_runtime_lease import digest
    return digest(path)

def _key(module,name,bound):
    suffix=''
    if name=='WaveEngine':
        source=bound.get('source') or Path(module.__file__).with_name('matrix_wave.h' if bound.get('matrix') else 'sediment_wave.h')
        suffix=sha(source)+':'+','.join(bound.get('names') or (('wave_state_gradients','matrix_bed_faces') if bound.get('matrix') else ('sediment_wave_faces',)))
    if name=='ObjectiveEngine':suffix=bound.get('gradient_kernel','flow2dh_objective_capacity.h')
    return module.__name__+'.'+name+':'+suffix

def _bindings(obj):
    result={'identity':obj.identity,'libraries':{},'functions':{}}
    for k,v in vars(obj).items():
        if isinstance(v,ctypes.CDLL):result['libraries'][k]=str(Path(v._name).resolve())
        elif isinstance(v,dict) and v and all(isinstance(f,ctypes._CFuncPtr) for f in v.values()):
            result['functions'][k]={n:f.__name__ for n,f in v.items()}
    return result

def _load_lib(path):
    with os.add_dll_directory(str(Path(path).parent)):
        return ctypes.CDLL(str(path))

def install(manifest=None,root=None):
    global ACTIVE
    if ACTIVE:raise RuntimeError('Portable river loaders already installed')
    ACTIVE=True
    recorded=manifest is None
    table=RECORDS if recorded else manifest
    root=Path(root or '.')
    def library(value):
        p=root/value['file']
        if sha(p)!=value['sha256']:raise ValueError('Portable library checksum differs')
        return p
    for module_name,name in ENGINE_CLASSES.items():
        module=importlib.import_module(module_name);cls=getattr(module,name);original=cls.__init__;signature=inspect.signature(original)
        def make(original,signature,module,name):
            @functools.wraps(original)
            def init(self,*args,**kwargs):
                bound=signature.bind(self,*args,**kwargs);bound.apply_defaults();b=bound.arguments
                if b.get('backend')!='cpu':return original(self,*args,**kwargs)
                threads=b.get('threads',4)
                if type(threads) is not int or not 1<=threads<=256:raise ValueError('Positive CPU thread count <=256 required')
                key=_key(module,name,b)
                if recorded:
                    original(self,*args,**kwargs);table[key]=_bindings(self);return
                if key not in table:raise ValueError('CPU kernel is not in this release: '+key)
                if name=='ClassicFlowEH' and (not np.isfinite([b['diameter'],b['chezy']]).all() or b['diameter']<=0 or b['chezy']<1.e-6):
                    raise ValueError('Invalid EH coefficients')
                row=table[key];self.backend='cpu';self.threads=threads;self.xp=np;self.dirs=[];self.dll_dirs=[]
                self.identity=json.loads(json.dumps(row['identity']));self.identity['threads']=threads
                for attr,item in row['libraries'].items():setattr(self,attr,_load_lib(library(item)))
                for attr,functions in row['functions'].items():
                    loaded={n:getattr(self.lib,symbol) for n,symbol in functions.items()}
                    for fn in loaded.values():fn.restype=None
                    setattr(self,attr,loaded)
                if name=='FlowEngine':self.local.braids_flow2dh_batch.restype=None
                if module.__name__ in ('braids_core.sediment.river_turbine_flow','braids_core.sediment.river_hydrokinetic_stress'):self.lib.hkt_threads=self.lib.hkt_stress_threads
                if name=='ClassicFlowEH':
                    self.d=float(b['diameter']);self.c=float(b['chezy'])
                    self.solid_density=2650.
                    self.identity.update(diameter_m=self.d,chezy=self.c)
                    self.lib.braids_flow_eh_cells.restype=self.lib.braids_flow_eh_slope.restype=None
            return init
        cls.__init__=make(original,signature,module,name)
    # Native hydraulic constructors load independent per-deck library instances.
    for module_name,func in (
        ('braids_core.sediment.river_turbine_flow','instrumented_native'),
        ('braids_core.sediment.river_mesh_records','build'),
        ('braids_core.sediment.bend_native_hydraulics','build'),
        ('braids_core.sediment.wall_bed_hydraulics','build'),
        ('braids_core.sediment.shoreline_solver','build_native'),
        ('validation.sediment.river_morphodynamics.shoreline_hydraulic_probe','build')):
        module=importlib.import_module(module_name);original=getattr(module,func)
        def make_build(module,func,original):
            def build(*args,**kwargs):
                key=module.__name__+'.'+func
                if 'shoreline_hydraulic_probe' in key:
                    base=importlib.import_module('braids_core.sediment.shoreline_solver').build_native()[0]
                    code=module.native_code((base.parent/'shoreline.f90').read_text(),args[0])
                    key+=':'+hashlib.sha256(code.encode()).hexdigest()
                if recorded:
                    result=original(*args,**kwargs)
                    table[key]={'native':str(result[0]),'identity':result[1]}
                    return result
                row=table[key];src=library(row['native'])
                cache=Path(os.environ['LOCALAPPDATA'])/'BRAIDS/native-river'/sha(src)
                cache.mkdir(parents=True,exist_ok=True)
                # NativeHydraulics creates per-case instances here, not in the installed payload.
                for file in src.parent.iterdir():
                    if file.is_file():
                        target=cache/file.name
                        if not target.exists():shutil.copy2(file,target)
                        if sha(target)!=sha(file):raise ValueError('Native cache checksum differs')
                return cache/src.name,row['identity'],str(cache/'precompiled-loader')
            return build
        setattr(module,func,make_build(module,func,original))
    columns=importlib.import_module('braids_core.sediment.bed_columns_v101');original=columns.native_runtime_v101
    @functools.lru_cache(maxsize=1)
    def column_runtime():
        key='columns_v101'
        if recorded:
            result=original();table[key]={'library':str(result[1]._name),'receipt':result[3]};return result
        row=table[key];dll=_load_lib(library(row['library']));fn=dll.braids_bed_columns_v101
        real=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');integer=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        fn.argtypes=([ctypes.c_int]*9+[ctypes.c_double]+[real]*7+[integer]+[ctypes.c_double]*3+[ctypes.c_int]+[real]*6+[integer]);fn.restype=None
        return fn,dll,None,row['receipt']
    columns.native_runtime_v101=column_runtime

