"""Stream the retained Iber GiD result prefix and compare raw cell fields.

No Iber file is modified; no new Iber simulation is launched. The supported
parser is deliberately narrow: this saved GiDPostEx1.2 little-endian file,
indexed scalar/vector results at one Gauss point per triangular cell.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import re
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

import prepare_matched134 as prep

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'validation/_performance134/iber_results'
REFERENCES=[
    'https://www.gidhome.com/ftp/pub/GiD_Documentation/Docs/GiD16/GiD_16_Customization_Manual.pdf',
    'https://www.gidhome.com/archive/GiD_Convention/2020/courses/gid_advanced/GiD_15_Advanced_Courses.pdf',
]
HEADER=re.compile(r'^Result "([^"]+)" "([^"]+)" (\S+) (Scalar|Vector) OnGaussPoints "GP_ELEMENT_1"$')
KEYS={'Elevation (m)':'bed','Depth (m)':'depth','Water Elevation (m)':'stage',
      'Velocity (m/s)':'velocity','Specific Discharge (m2/s)':'discharge','Froude':'froude',
      'Froude number':'froude','Froude Number':'froude'}


def exact(stream,n):
    b=stream.read(n)
    if len(b)!=n:raise ValueError('Truncated GiD result record')
    return b


def integer(stream):
    return struct.unpack('<i',exact(stream,4))[0]


def string(stream):
    n=integer(stream)
    if not 1<=n<=100000:raise ValueError('Invalid GiD string length '+str(n))
    b=exact(stream,n)
    if b[-1]!=0:raise ValueError('Unterminated GiD string')
    return b[:-1].decode('ascii')


def read_values(stream,components,ncells):
    result=np.full((ncells,components),np.nan)
    seen=np.zeros(ncells,dtype=bool)
    while True:
        cell=integer(stream)
        if cell==-1:break
        if not 1<=cell<=ncells or seen[cell-1]:raise ValueError('Invalid or repeated GiD cell ID')
        v=np.frombuffer(exact(stream,4*components),dtype='<f4')
        if not np.isfinite(v).all():raise ValueError('Nonfinite GiD value')
        result[cell-1]=v;seen[cell-1]=True
    if string(stream)!='End Values':raise ValueError('Invalid GiD value terminator')
    return result,seen


def parser_self_test():
    def end():
        value=b'End Values\x00'
        return struct.pack('<ii',-1,len(value))+value
    scalar=struct.pack('<if',2,.25)+end()
    values,mask=read_values(io.BytesIO(scalar),1,3)
    assert mask.tolist()==[False,True,False] and values[1,0]==.25
    vector=struct.pack('<i4f',1,3.,4.,0.,5.)+end()
    values,mask=read_values(io.BytesIO(vector),4,2)
    assert mask.tolist()==[True,False] and np.array_equal(values[0],[3,4,0,5])
    invalid=[struct.pack('<if',0,1.)+end(),struct.pack('<if',4,1.)+end(),
             struct.pack('<ifif',1,1.,1,2.)+end(),struct.pack('<if',1,float('nan'))+end(),
             scalar[:-1]]
    for data in invalid:
        try:read_values(io.BytesIO(data),1,3)
        except ValueError:pass
        else:raise AssertionError('Corrupt parser fixture accepted')
    return dict(valid_scalar=True,valid_vector=True,rejected_corrupt_records=len(invalid))


def extract(targets=(0.,10.,30.,60.)):
    OUT.mkdir(parents=True,exist_ok=True)
    nodes,cells,_=prep.mesh();path=prep.IBER/'test.post.res'
    snapshots={};bed=None;headers=[];parsed=0;last_time=0.
    with gzip.open(path,'rb') as stream:
        if integer(stream)!=0x91d:raise ValueError('Unexpected GiD endian/magic marker')
        if string(stream)!='GiDPostEx1.2':raise ValueError('Unsupported GiD result version')
        header=None
        while True:
            text=string(stream)
            if text.startswith('Result '):
                match=HEADER.fullmatch(text)
                if not match:raise ValueError('Unsupported GiD result header: '+text)
                name,analysis,clock,kind=match.groups();time=float(clock)
                if time>max(targets)+1.:
                    last_time=time;break
                header=(name,analysis,time,kind);headers.append(text)
            elif text.startswith('Values'):
                if text!='Values -1 Indexed' or header is None:
                    raise ValueError('Unsupported non-indexed/unbound GiD values')
                name,analysis,time,kind=header
                # GiD vector storage contains x, y, z and its modulus, as
                # documented in the official GiD advanced-course guide.
                values,present=read_values(stream,1 if kind=='Scalar' else 4,len(cells))
                parsed+=1
                if name=='Elevation (m)':
                    if not present.all():raise ValueError('Iber topography omits cells')
                    bed=values[:,0]
                elif analysis=='Hydraulic':
                    snapshots.setdefault(time,{})[name]=(values,present)
                header=None
            elif text.startswith(('ResultRangesTable ','End ResultRangesTable','ComponentNames ',
                                  ' 0 ',' 0.5 ','1 -','1 -',' 1 -','86400 ',' 7200 ')):
                pass
            else:
                # Before data, ordinary range-table labels are metadata. No
                # unknown structure is accepted after a live result header.
                if header is not None:raise ValueError('Unsupported GiD header metadata: '+text)
        position=stream.tell()
    if bed is None:raise ValueError('No Iber bed field')
    available=sorted(snapshots)
    log_times=[float(m.group(1)) for m in re.finditer(r'^\s*([\d.]+)\s+[\d.]+\s+\d+:\d+:\d+:\d+',
                (prep.IBER/'proceso.rep').read_text(),flags=re.M)]
    output={'nodes':nodes,'cells':cells,'bed':bed};metadata=[]
    for target in targets:
        t=min(available,key=lambda v:abs(v-target));row=snapshots[t]
        required=('Depth (m)','Water Elevation (m)','Velocity (m/s)','Specific Discharge (m2/s)')
        if not all(k in row for k in required):raise ValueError('Incomplete selected Iber time '+str(t))
        depth,mask=row['Depth (m)'];depth=np.nan_to_num(depth[:,0],nan=0.)
        velocity=row['Velocity (m/s)'][0];discharge=row['Specific Discharge (m2/s)'][0]
        vm=row['Velocity (m/s)'][1];qm=row['Specific Discharge (m2/s)'][1]
        if not np.array_equal(mask,vm) or not np.array_equal(mask,qm):raise ValueError('Iber variable masks differ')
        velocity=np.nan_to_num(velocity,nan=0.);discharge=np.nan_to_num(discharge,nan=0.)
        stage=row['Water Elevation (m)'][0][:,0]
        vector_error=np.max(abs(np.linalg.norm(velocity[:,:3],axis=1)-velocity[:,3]))
        q_error=np.max(abs(discharge[:,:2]-velocity[:,:2]*depth[:,None]))
        stage_error=np.max(abs(stage[mask]-bed[mask]-depth[mask])) if mask.any() else 0.
        if vector_error>1e-5 or stage_error>1e-4:
            raise ValueError(('Iber field consistency failed',vector_error,q_error,stage_error))
        discharge_norm_error=np.max(abs(np.linalg.norm(discharge[:,:3],axis=1)-discharge[:,3]))
        if discharge_norm_error>1e-5:raise ValueError('Iber discharge vector modulus differs')
        q_index=int(np.argmax(np.max(abs(discharge[:,:2]-velocity[:,:2]*depth[:,None]),axis=1)))
        key='t'+str(int(target));output[key+'_depth']=depth;output[key+'_velocity']=velocity[:,:2]
        output[key+'_discharge']=discharge[:,:2]
        output[key+'_stage']=stage;output[key+'_present']=mask
        logged=min(log_times,key=lambda v:abs(v-t))
        if abs(logged-t)>.051:raise ValueError('Cannot map rounded GiD time to Iber runtime log')
        metadata.append(dict(target_s=target,saved_time_s=t,logged_time_s=logged,key=key,wet_cells=int(mask.sum()),
            maximum_depth_m=float(depth.max()),maximum_speed_m_s=float(velocity[:,3].max()),
            vector_modulus_error_m_s=float(vector_error),discharge_hu_error_m2_s=float(q_error),
            discharge_modulus_error_m2_s=float(discharge_norm_error),
            maximum_discharge_hu_gap_cell_1based=q_index+1,
            maximum_discharge_hu_gap_depth_m=float(depth[q_index]),
            maximum_discharge_hu_gap_velocity=velocity[q_index,:].tolist(),
            maximum_discharge_hu_gap_discharge=discharge[q_index,:].tolist(),
            stage_bed_plus_depth_error_m=float(stage_error)))
    output['times']=np.array([r['saved_time_s'] for r in metadata])
    target=OUT/'IBER_EARLY_FIELDS.npz';np.savez_compressed(target,**output)
    report=dict(source=str(path),source_sha256=prep.sha(path),output=str(target),output_sha256=prep.sha(target),
                parser_sha256=prep.sha(__file__),official_format_references=REFERENCES,
                format='GiDPostEx1.2 gzip; little-endian int32 and float32; one Gauss point per cell',
                parsed_result_records=parsed,decompressed_bytes_consumed=position,stopped_at_first_header_time_s=last_time,
                parser_unit_checks=parser_self_test(),
                time_log_source=str(prep.IBER/'proceso.rep'),time_log_sha256=prep.sha(prep.IBER/'proceso.rep'),
                available_saved_times_s=available,selected=metadata,
                saved_time_precision_note='GiD headers round time to one decimal here. Selected nearest log rows provide millisecond-resolution printed physical times; both retained.',
                encountered_result_names=sorted({HEADER.fullmatch(h).group(1) for h in headers}),
                dry_policy='Saved Iber configuration omits dry elements. Omitted depth and velocity assigned zero only for plotting/volume; stage remains NaN and explicit present mask retained.',
                bed_vs_vertex_mean_rms_m=float(np.sqrt(np.mean((bed-nodes[cells,2].mean(1))**2))),
                note='Parser checked against official format description, vector moduli, water-level and bed identities. Saved Iber discharge is NOT h times saved velocity in every cell; this discrepancy is recorded, not treated as a passed consistency test or corrected away. Plots use the separately reported depth/velocity. This is not independent GiD software verification.')
    (OUT/'EXTRACTION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['selected'],indent=2))
    return report


def load_braids(run,target):
    root=Path(run)/'frames';index=json.loads((root/'INDEX.json').read_text())
    row=min(index['frames'],key=lambda r:abs(r['time_s']-target))
    fp=root/row['file'];mp=root/row['epoch']
    if prep.sha(fp)!=row['sha256'] or prep.sha(mp)!=index['epochs'][row['epoch']]:raise ValueError('BRAIDS frame checksum differs')
    with np.load(fp,allow_pickle=False) as f:q=f['q'].copy();z=f['z'].copy();area=f['area'].copy()
    with np.load(mp,allow_pickle=False) as f:
        polygons=[f['vertices'][i:j].copy() for i,j in zip(f['offsets'][:-1],f['offsets'][1:])]
    u=np.divide(q[1:],q[0],out=np.zeros_like(q[1:]),where=q[0]>1e-6)
    return dict(time=row['time_s'],polygons=polygons,depth=q[0],speed=np.linalg.norm(u,axis=0),bed=z,
                volume=float(q[0]@area),wet_area=float(area[q[0]>.001].sum()))


def compare(run):
    report=json.loads((OUT/'EXTRACTION.json').read_text())
    with np.load(OUT/'IBER_EARLY_FIELDS.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    nodes=data['nodes'];cells=data['cells'];origin=nodes[:,:2].min(0)
    polygons=nodes[cells,:2]-origin
    # Plot raw cell values on each native mesh, no interpolated smooth picture.
    targets=(10,30,60);both=[];metrics=[]
    for t in targets:
        r=next(r for r in report['selected'] if r['target_s']==t);key=r['key']
        b=load_braids(run,t);h=data[key+'_depth'];speed=np.linalg.norm(data[key+'_velocity'],axis=1)
        from shapely.geometry import Polygon
        iarea=np.array([Polygon(p).area for p in polygons])
        metrics.append(dict(target_s=t,iber_saved_s=r['saved_time_s'],iber_logged_s=r['logged_time_s'],braids_saved_s=b['time'],
            iber_volume_m3=float(h@iarea),braids_volume_m3=b['volume'],
            iber_wet_area_1mm_m2=float(iarea[h>.001].sum()),braids_wet_area_1mm_m2=b['wet_area'],
            iber_maximum_depth_m=float(h.max()),braids_maximum_depth_m=float(b['depth'].max()),
            iber_maximum_speed_m_s=float(speed.max()),braids_maximum_speed_m_s=float(b['speed'].max())))
        both.append((r,b,h,speed))
    wet_points=[]
    for _,b,h,_ in both:
        wet_points += [polygons[i] for i in np.flatnonzero(h>=.001)]
        wet_points += [b['polygons'][i]-origin for i in np.flatnonzero(b['depth']>=.001)]
    wet_points=np.vstack(wet_points);lo=wet_points.min(0)-6;hi=wet_points.max(0)+6
    from shapely.ops import unary_union
    iber_outline=np.array(unary_union([Polygon(p) for p in polygons]).exterior.coords)
    braids_outline=np.array(unary_union([Polygon(p-origin) for p in both[-1][1]['polygons']]).exterior.coords)
    for quantity,label,cmap in [('depth','Water depth (m)','Blues'),('speed','Speed (m/s)','Reds')]:
        vmax=max(max(float(a[1][quantity].max()),float((a[2] if quantity=='depth' else a[3]).max())) for a in both)
        norm=Normalize(0,vmax)
        fig,axes=plt.subplots(2,3,figsize=(11,9),layout='constrained')
        for col,(r,b,h,speed) in enumerate(both):
            iber=h if quantity=='depth' else speed
            for rr,polys,values,depth,time in [(0,polygons,iber,h,r['logged_time_s']),
                (1,[p-origin for p in b['polygons']],b[quantity],b['depth'],b['time'])]:
                ax=axes[rr,col];values=np.ma.masked_where(depth<.001,values)
                artist=PolyCollection(polys,array=values,cmap=cmap,norm=norm,edgecolors='none')
                ax.add_collection(artist);ax.plot(*(iber_outline if rr==0 else braids_outline).T,color='#888888',linewidth=.6)
                ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal',title=('Iber' if rr==0 else 'BRAIDS')+f' t={time:g} s')
                ax.set_xlabel('Relative easting (m)');ax.set_ylabel('Relative northing (m)')
        fig.colorbar(artist,ax=axes.ravel().tolist(),label=label,shrink=.8)
        fig.suptitle(label+' — wet-front detail on raw native cells; depths below 1 mm transparent\nComparable reach, not identical boundaries/discretization; no accuracy-pass claim',fontsize=11)
        fig.savefig(OUT/('IBER_BRAIDS_'+quantity.upper()+'.png'),dpi=160);plt.close(fig)
    result=dict(braids_run=str(Path(run).resolve()),metrics=metrics,
                limitations='Axis alignment: inlet39.1648→35.75 m (-8.72%); outlet30.9233→29.25 m (-5.41%). Domain symmetric change0.444%; reconstructed-bed rasterization sample1.95mm RMS/49.2mm max. Different dry/wet, wall friction, boundary and spatial methods remain. No formal accuracy gate inferred from these maps.')
    (OUT/'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');parser.add_argument('--braids')
    args=parser.parse_args()
    if args.extract or not (OUT/'EXTRACTION.json').exists():extract()
    if args.braids:compare(args.braids)


if __name__=='__main__':main()
