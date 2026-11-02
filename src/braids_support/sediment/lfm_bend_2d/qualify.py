"""Short-run operational gates, explicitly distinct from laboratory braids_support."""
import json
import numpy as np
from netCDF4 import Dataset
from . import run_case as r
from . import analyze_runs as a
from . import experiment as e


def acceleration(reference, accelerated):
    np.testing.assert_allclose(reference['xy'],accelerated['xy'],atol=1e-12)
    if not np.isclose(reference['th'][-1],accelerated['th'][-1]):
        raise ValueError('Acceleration control requires identical morphological time')
    mask=(reference['s']>=e.APPROACH)&(reference['s']<=e.BEND_END)
    base=reference['z'][-1]-reference['z'][0]
    delta=accelerated['z'][-1]-accelerated['z'][0]-base
    weights=reference['area'][mask]
    rms=float(np.sqrt(np.average(delta[mask]**2,weights=weights)))
    signal=float(np.sqrt(np.average(base[mask]**2,weights=weights)))
    relative=rms/max(signal,1e-12)
    return dict(factor=accelerated['p']['morfac'],rms_bed_difference_m=rms,
        reference_rms_bed_change_m=signal,relative_rms_difference=relative,
        passed=rms<=.0005 and relative<=.10,
        criterion='Both <=0.5 mm RMS and <=10% relative RMS over bend; a short-screen gate, not long-run equivalence',
        includes_secondary_reinitialization=True)


def capacity_audit(native,delft):
    folder=native['folder'];rows=np.loadtxt(folder/'secondary_flow.txt')
    rows=rows[np.isclose(rows[:,0],native['t'][-1])];ids=rows[:,1].astype(int)
    keep=(native['s'][ids]>2)&(native['s'][ids]<e.LENGTH-2)
    magnitude=np.hypot(rows[:,8],rows[:,9])
    # Diagnostic is potential capacity per material, before multiplying by
    # the 50/50 active-layer shares. It must NOT be summed as accepted flux.
    expected=r.RHOS*r.mpm_capacity(native['h'][-1,ids],native['speed'][-1,ids])
    native_error=float(np.max(abs(magnitude[keep]-expected[keep])/np.maximum(expected[keep],1e-30)))
    with Dataset(delft['folder']/'trim-basin.nc') as d:
        extra=np.asarray(d['excess_theta001'][-1,1:-1,1:-1])
        expected=8*np.sqrt(9.81*1.65*r.D50**3)*np.maximum(extra,0)**1.5
        magnitude=np.hypot(d['SBCU'][-1,0,1:-1,1:-1],d['SBCV'][-1,0,1:-1,1:-1])
        delft_error=float(np.max(abs(magnitude-expected)/np.maximum(expected,1e-30)))
        h=np.asarray(d['S1'][-1,1:-1,1:-1]+d['DPS'][-1,1:-1,1:-1])
        speed=np.asarray(d['UMOD'][-1,1:-1,1:-1])
        predicted=r.mpm_capacity(h,speed)
        collocated_error=float(np.max(abs(magnitude-predicted)/np.maximum(predicted,1e-30)))
    return dict(native_potential_capacity_relative_error=native_error,
        delft_capacity_vs_reported_excess_shields_relative_error=delft_error,
        delft_capacity_vs_cell_depth_roughness_relative_error=collocated_error,
        passed=native_error<1e-10 and delft_error<1e-10 and collocated_error<.01,
        note='Execution diagnostics of unchanged engines. Cell/face roughness interpolation is not identical; this does not prove full closure equivalence to the smoothed 2DEF law.')


def run():
    baseline={engine:a.load(r.HERE/'runs'/name) for engine,name in dict(
        braids='mobile120_m1_b_braids',delft='mobile120_m1_d_delft').items()}
    hydro=[a.load(r.HERE/'runs'/f'hydraulic_v1_{engine}') for engine in baseline]
    records=[];passed=True
    for engine,base in baseline.items():
        control=a.audit(base)
        flow=a.audit(next(d for d in hydro if d['p']['engine']==engine))
        gates=dict(steady_hydraulics=flow['final_stage_change_over_output_interval_m']<1e-5,
            inlet_Q=flow['inlet_discharge_relative_error']<.005,
            outlet_Q=flow['outlet_discharge_relative_error']<.005,
            hydro_water_balance=abs(flow['water_budget_residual_m3'])<1e-6,
            full_sediment_feed=abs(control['prescribed_feed_fraction']-1)<.005,
            sediment_budget=control['sediment_budget_residual_over_throughput']<.01,
            sediment_leaves_domain=control['exported_sediment_kg']>0,
            depth_averaged=(base['nk']==1 if engine=='delft' else True))
        controls=[]
        for factor,tag in [(10,'mobile120_m10_b' if engine=='braids' else 'mobile120_m10_d'),(5,'mobile120_m5')]:
            controls.append(acceleration(base,a.load(r.HERE/'runs'/f'{tag}_{engine}')))
        passed &= all(gates.values())
        records.append(dict(engine=engine,gates=gates,hydraulic=flow,mobile=control,acceleration=controls))
    capacity=capacity_audit(baseline['braids'],baseline['delft'])
    result=dict(operational_pair_passed=bool(passed and capacity['passed']),cases=records,capacity=capacity,
        factor5_short_screen_passed=all(row['acceleration'][1]['passed'] for row in records),
        laboratory_validation_passed=False,grid_convergence_established=False,
        publication_inputs_exactly_reproduced=False,
        unresolved=['full-time acceleration sensitivity','mesh convergence','boundary-distance sensitivity',
                    'experimental equilibrium','exact 2DEF smoothing and secondary-flow closure differences'])
    r.write_json(r.HERE/'reports/qualification_v1/metrics.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':run()
