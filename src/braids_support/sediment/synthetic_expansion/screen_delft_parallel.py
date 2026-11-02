"""Serial/MPI control using the locked, unmodified Delft3D executable and inputs.

The official run_dflow2d3d_parallel.sh uses mpirun -np N d_hydro config.xml.
The minimal locked image lacks the launcher, so the already installed Intel MPI
2021.13 tools are mounted read-only from a dedicated Docker volume. No source,
engine binary or library in /delft3d is overlaid or changed.
"""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import numpy as np
import run_comparison as case


def execute(folder, ranks, timeout=1800):
    if not isinstance(ranks,int) or not 2 <= ranks <= 16:
        raise ValueError('Diagnostic MPI ranks must be in 2..16')
    lock = case.delft_runtime.verified_runtime()
    name = 'braids-mpi-screen-'+uuid.uuid4().hex[:12]
    docker = ['wsl.exe','-d',lock['wsl_distribution'],'--','docker']
    tools_lock=json.loads((case.HERE/'delft_mpi_tools.lock.json').read_text())
    mount='type=volume,source='+tools_lock['docker_volume']+',target=/mpi,readonly'
    hashes=subprocess.run(docker+['run','--rm','--mount',mount,'--entrypoint',
        '/usr/bin/sha256sum',lock['image_reference'],*tools_lock['sha256']],
        text=True,capture_output=True,check=True,timeout=60)
    observed_hashes={line.split()[1]:line.split()[0] for line in hashes.stdout.splitlines()}
    if observed_hashes!=tools_lock['sha256']:
        raise ValueError('MPI auxiliary tools or locked engine library identity changed')
    command = docker+['run','--rm','--name',name,'--shm-size','1g',
        '--mount',mount,
        '-v',f'{case.delft_runtime.windows_to_wsl(folder)}:/job','-w','/job',
        '-e','I_MPI_ROOT=/mpi','-e','I_MPI_FABRICS=shm','-e','I_MPI_PIN=0',
        '--entrypoint','/mpi/bin/mpiexec.hydra',lock['image_reference'],
        '-n',str(ranks),lock['executable'],'config_d_hydro.xml']
    case.write(folder/'launch.json',json.dumps(dict(command=command,container_name=name,
        shared_memory_capacity='1 GiB',purpose='Runtime configuration control; no engine or source changes'),indent=2))
    started = time.perf_counter()
    try:
        run = subprocess.run(command,text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(docker+['rm','--force',name],capture_output=True,check=False)
        raise
    case.write(folder/'delft3d.log',run.stdout+run.stderr)
    observed = sorted(set(map(int,re.findall(r'MPI process number\s+(\d+)',run.stdout))))
    result = dict(command=command,return_code=run.returncode,
        elapsed_wall_s=time.perf_counter()-started,ranks=ranks,runtime=lock,
        runtime_and_auxiliary_sha256=observed_hashes,
        observed_process_numbers=observed,
        success=(run.returncode==0 and 'shutting down normally' in run.stdout and
                 (folder/'trim-basin.nc').exists() and observed==list(range(ranks))),
        auxiliary_launcher='Read-only Intel MPI 2021.13 tools; original image /delft3d tree unchanged')
    case.write(folder/'execution.json',json.dumps(result,indent=2))
    if not result['success']:
        raise RuntimeError(f'MPI screen failed: {folder}/delft3d.log')
    return result


def compare_outputs(source, folder, report):
    """Strict trajectory comparison; execution and time/mesh identity required."""
    from qualify_bank_resolved_timestep import require_successful_execution
    for path in (source, folder):
        require_successful_execution(path)
    plan = json.loads((report/'plan.json').read_text())
    execution = json.loads((folder/'execution.json').read_text())
    a,b = [case.common.inspect_delft3d(path) for path in (source,folder)]
    for group,fields in (('map',('time','KCS','XZ','YZ','GSQS')),('history',('time',))):
        for key in fields:
            if not np.array_equal(a[group][key],b[group][key]):
                raise ValueError(f'MPI comparison requires identical {group}/{key}')
    differences = {}
    for key in ('DPS','S1','U1','V1'):
        left,right=a['map'][key],b['map'][key]
        if left.shape!=right.shape or not np.isfinite(left-right).all():
            raise ValueError(f'Invalid or incomplete MPI field {key}')
        differences[key]=float(np.max(abs(left-right)))
    aa,bb = a['history']['SBTRC'],b['history']['SBTRC']
    if aa.shape!=bb.shape or not np.isfinite(aa-bb).all():
        raise ValueError('Invalid or incomplete MPI sediment ledger')
    relative = float(np.max(abs(aa-bb))/max(float(np.max(abs(aa))),1e-30))
    limit=plan['limits']
    passed = differences['DPS']<=limit['max_bed_difference_m'] and differences['S1']<=limit['max_water_surface_difference_m'] and max(differences['U1'],differences['V1'])<=limit['max_velocity_difference_m_s'] and relative<=limit['relative_cumulative_sediment_difference']
    result = dict(execution=execution,max_absolute_differences=differences,
        relative_cumulative_sediment_difference=relative,passed=passed,
        serial_wall_s=json.loads((source/'execution.json').read_text())['elapsed_wall_s'])
    case.write(report/'metrics.json',json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=case.HERE/'runs/bank_outlet_mobile_coarse_v1/dx1/delft3d')
    p.add_argument('--name',default='bank_delft_parallel_v1')
    p.add_argument('--subdir',default='',help='Optional case subdirectory within the versioned run')
    p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--defer-comparison',action='store_true',help='Execute unchanged inputs only; serial comparison must be done separately')
    args = p.parse_args()
    if Path(args.subdir).is_absolute() or '..' in Path(args.subdir).parts:
        raise ValueError('Case subdirectory must stay within the versioned run')
    folder = case.HERE/'runs'/args.name/args.subdir
    report = case.HERE/'reports'/args.name
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    original = json.loads((args.source/'input_provenance.json').read_text())
    for filename,digest in original['files'].items():
        if case.sha256(args.source/filename)!=digest:
            raise ValueError(f'Input differs from recorded source: {filename}')
        shutil.copy2(args.source/filename,folder/filename)
    shutil.copy2(args.source/'input_provenance.json',folder/'input_provenance.json')
    plan = dict(source=str(args.source),ranks=args.ranks,identical_inputs=original['files'],
        limits={'max_water_surface_difference_m':1e-8,'max_bed_difference_m':1e-8,
                'max_velocity_difference_m_s':1e-7,'relative_cumulative_sediment_difference':1e-6},
        prohibition='No Delft3D source, engine or original library changes')
    case.write(report/'plan.json',json.dumps(plan,indent=2))
    execution = execute(folder,args.ranks,args.timeout)
    if args.defer_comparison:
        case.write(report/'pending_comparison.json',json.dumps(dict(execution=execution,
            serial_comparison_passed=False,reason='Deferred explicitly; not yet qualified'),indent=2))
        print(json.dumps(execution,indent=2),flush=True)
        return
    result = compare_outputs(args.source,folder,report)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
