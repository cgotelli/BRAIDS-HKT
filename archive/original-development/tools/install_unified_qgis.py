"""Recoverable default-profile installation of an authenticated unified ZIP."""
import argparse
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import tempfile
import zipfile
from build_unified_qgis import verify,sha,write


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();archive=a.archive.resolve()
    processes=subprocess.run(['tasklist.exe','/FO','CSV','/NH'],capture_output=True,text=True,check=True)
    if any(row and 'qgis' in row[0].lower() for row in csv.reader(processes.stdout.splitlines())):
        raise RuntimeError('Close QGIS before replacing the installed plugin')
    parent=(Path(os.environ['APPDATA'])/'QGIS/QGIS3/profiles/default/python/plugins').resolve()
    target=(parent/'braids_qgis').resolve()
    if target.parent!=parent or target.name!='braids_qgis':raise ValueError('Invalid installation target')
    parent.mkdir(parents=True,exist_ok=True)
    backup=parent/'.braids_release_backups'/('braids_qgis_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with tempfile.TemporaryDirectory(prefix='.braids_unified_',dir=parent) as temporary:
        stage=Path(temporary)
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
            for name in names:
                parts=PurePosixPath(name)
                if ('\\' in name or parts.is_absolute() or '..' in parts.parts or
                        not parts.parts or parts.parts[0]!='braids_qgis'):
                    raise ValueError('Unsafe archive member')
            z.extractall(stage)
        payload=stage/'braids_qgis';manifest=verify(payload)
        moved=False
        try:
            if target.exists():
                backup.parent.mkdir(parents=True,exist_ok=True);os.replace(target,backup);moved=True
            os.replace(payload,target)
            verify(target)
        except BaseException:
            if moved:
                if target.exists():os.replace(target,stage/'failed_payload')
                os.replace(backup,target)
            raise
    report=dict(status='installed',version=manifest['version'],destination=str(target),
                archive=str(archive),archive_sha256=sha(archive),
                backup=str(backup) if moved else None,files_verified=len(manifest['files']))
    write(a.report,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
