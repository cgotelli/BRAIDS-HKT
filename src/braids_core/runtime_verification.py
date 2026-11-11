"""Standalone authenticated bootstrap: no imports from the application.

Every path/checksum is checked. NTFS hard-link aliases can reuse a digest only
while its underlying file is held open denying writes and deletion. No mtime,
size, persisted receipt or claimed manifest checksum substitutes for file bytes.
"""
import collections
import ctypes
import hashlib
import os


def verify_files(root,files):
    entries=[];counts=collections.Counter();held={};hashed=hits=0
    for name,expected in files.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('Runtime artifact differs: '+name)
        key=None
        if os.name=='nt' and path.suffix.lower()=='.dll':
            state=path.stat()
            if state.st_ino:key=(state.st_dev,state.st_ino);counts[key]+=1
        entries.append((name,path,expected,key))
    try:
        for name,path,expected,key in entries:
            if key is None or counts[key]<2:
                raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest();hashed+=len(raw)
            else:
                import msvcrt
                kernel=ctypes.WinDLL('kernel32',use_last_error=True)
                create=kernel.CreateFileW
                create.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]
                create.restype=ctypes.c_void_p
                handle=create(os.fspath(path),0x80000000,1,None,3,0x80,None)
                if handle==ctypes.c_void_p(-1).value:raise ctypes.WinError(ctypes.get_last_error())
                try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
                except BaseException:
                    kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.CloseHandle(handle);raise
                stream=os.fdopen(descriptor,'rb')
                try:
                    state=os.fstat(stream.fileno());actual_id=(state.st_dev,state.st_ino)
                    if state.st_ino and actual_id in held:
                        actual=held[actual_id][1];hits+=1
                    else:
                        actual=hashlib.file_digest(stream,'sha256').hexdigest();hashed+=state.st_size
                        if state.st_ino:held[actual_id]=(stream,actual);stream=None
                finally:
                    if stream is not None:stream.close()
            if actual!=expected:raise ValueError('Runtime artifact differs: '+name)
        return dict(files=len(entries),bytes_hashed=hashed,locked_alias_hits=hits,
                    locked_file_groups=len(held))
    finally:
        for stream,_ in held.values():stream.close()
