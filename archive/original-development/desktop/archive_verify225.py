"""Read-only full-content verification before moving historical artifacts to D:."""
import hashlib,json,os,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def extended(path):
    text=str(path.absolute())
    return Path('\\\\?\\'+text) if os.name=='nt' and not text.startswith('\\\\?\\') else path

def inventory(root):
    root=extended(root)
    result={};stack=[root]
    while stack:
        folder=stack.pop()
        with os.scandir(folder) as entries:
            for e in entries:
                st=e.stat(follow_symlinks=False)
                if getattr(st,'st_file_attributes',0)&0x400:raise ValueError('Reparse point inside archive: '+e.path)
                if e.is_dir(follow_symlinks=False):stack.append(Path(e.path))
                elif e.is_file(follow_symlinks=False):result[Path(e.path).relative_to(root).as_posix()]=st.st_size
    return result

def digest(path):
    h=hashlib.sha256()
    with extended(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    source,target,receipt=map(Path,sys.argv[1:])
    workspace=Path(__file__).resolve().parents[1]
    assert source.resolve().is_relative_to(workspace) and source.resolve()!=workspace
    assert target.resolve().is_relative_to(Path('D:/BRAIDS_archive/20261030_cleanup225'))
    a,b=inventory(source),inventory(target)
    prior=json.loads(receipt.read_text()) if receipt.exists() else None
    recovering=a!=b
    if recovering:
        assert prior and prior['all_file_sha256_equal'] and prior['source']==str(source.resolve()) and prior['destination']==str(target.resolve())
        assert all(n in b and b[n]==size for n,size in a.items()),'Residual source is not a subset of the verified archive'
    h=hashlib.sha256();start=time.monotonic()
    def compare(n):
        x=digest(target/n)
        if n in a:assert x==digest(source/n),n
        return n,x
    names=sorted(b);done=0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for begin in range(0,len(names),2048):
            for n,x in pool.map(compare,names[begin:begin+2048]):
                h.update((n+'\0'+str(b[n])+'\0'+x+'\n').encode('utf8'));done+=1
            if done//25000!=(done-2048)//25000:print(source.name,'verified',done,'/',len(b),flush=True)
    if recovering:assert h.hexdigest()==prior['inventory_sha256'],'Original archive content changed'
    result=dict(source=str(source.resolve()),destination=str(target.resolve()),files=len(b),bytes=sum(b.values()),
        inventory_sha256=h.hexdigest(),all_file_sha256_equal=True,seconds=time.monotonic()-start)
    if recovering:result['residual_source_files_before_completion']=len(a)
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Verified',source.name,len(b),'files',round(sum(b.values())/2**30,3),'GiB',flush=True)

if __name__=='__main__':main()
