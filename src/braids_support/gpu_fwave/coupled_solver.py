"""Explicit CUDA fixed-grid MPM/secondary-flow morphodynamic solver.

This joins resident numerical operators; no per-stage field is evaluated on CPU.
Preparation, I/O and convergence/timestep control remain host responsibilities.
Scope is validated explicitly, never reduced to fit an available implementation.
"""
from pathlib import Path
import numpy as np
from .reconstruction import ResidentReconstruction
from .hydraulic_pipeline import ResidentFlux,boundary_data
from .boundaries import BoundaryControls
from .sediment_transport import ResidentSedimentTransport
from .secondary_transport import SecondaryTransport,validate_scope as validate_secondary_scope
from .exner_update import ResidentBedUpdate
from .coupled_time import CoupledRK33,CoupledRK43
from .suspended_faces import ResidentSuspendedFaces
from .suspended_advection import ResidentSuspendedAdvection
from .suspended_exchange import ResidentNoncohesiveExchange
from .exchange_ledger_parallel import ResidentExchangeLedger

HERE=Path(__file__).resolve().parent


def normalized(data):
    data=dict(data)
    extensions=[]
    for key in ('outlet_extension','cpu_outlet_extension'):
        if key in data:
            value=np.asarray(data[key])
            if value.shape!=() or value.dtype.kind not in 'iu' or int(value) not in (0,1):
                raise ValueError('Coupled GPU outlet extension must be 0 or 1')
            extensions.append(int(value))
    if extensions and len(set(extensions))!=1:
        raise ValueError('Prepared CPU/GPU outlet extension controls disagree')
    data['outlet_extension']=extensions[0] if extensions else 0
    aliases={'dry':'dry_tolerance','cfl_target':'cfl','manning_cell':'manning',
             'layer_mass':'layers','rock':'bedrock','morfac':'morphological_factor'}
    for target,source in aliases.items():
        if target not in data and source in data:data[target]=data[source]
    for key in ('dry','grav','cfl_target','speed_limit','dx','dy'):
        if key not in data or np.asarray(data[key]).size!=1 or not np.isfinite(data[key]).all():
            raise ValueError('Missing/invalid coupled control '+key)
        data[key]=float(np.asarray(data[key]).item())
    data['dt']=float(np.asarray(data.get('dt',data.get('dtmax',.001))).item())
    secondary_model=int(np.asarray(data['secondary_flow_model']).item())
    secondary_index=int(np.asarray(data.get('secondary_index',-1)).item())
    if secondary_model==2:
        if 'state' in data and 'secondary_index' in data:
            if secondary_index<3:raise ValueError('Prognostic secondary state is required')
            data['secondary_mass']=np.ascontiguousarray(data['state'][secondary_index])
    elif secondary_model in (0,1,3):
        if secondary_index!=-1:raise ValueError('Stateless secondary modes cannot contain a prognostic secondary state')
        if 'secondary_mass' not in data:data['secondary_mass']=np.zeros(np.asarray(data['q']).shape[1],dtype=np.float64)
    if 'state' in data and 'suspended_indices' in data:
        indices=np.asarray(data['suspended_indices'])
        state=np.asarray(data['state'])
        if (indices.ndim!=1 or indices.dtype.kind not in 'iu' or state.ndim!=2 or
                np.any(indices<0) or np.any(indices>=state.shape[0]) or
                len(np.unique(indices))!=len(indices)):
            raise ValueError('Invalid suspended state indices')
        data['suspended_mass']=np.ascontiguousarray(state[indices.astype(int)])
    return data


def validate_scope(data, *, scheduled=False, bedform_variant=False,
                   general_avalanche_variant=False,
                   physical_toe_mode2_variant=False,
                   bank_avalanche_mode2_variant=False,
                   g3_bank_secondary_variant=False,
                   distributed_bank_mode3_variant=False,
                   distributed_bank_avalanche_mode3_variant=False,
                   recursive_bedform_bank_avalanche_mode2_variant=False,
                   recursive_bedform_bank_avalanche_mode3_variant=False,
                   bedload_adaptation_variant=False, evolving_toe_variant=False,
                   external_boundary_schedule=False, external_level_secondary=False,
                   external_physical_toe=False,
                   experimental_morfac_mode3=False):
    """Reject missing identity and unported processes before initializing CUDA."""
    combined_bedform_mode2 = bool(
        recursive_bedform_bank_avalanche_mode2_variant)
    combined_bedform_mode3 = bool(
        recursive_bedform_bank_avalanche_mode3_variant)
    combined_bedform_bank = combined_bedform_mode2 or combined_bedform_mode3
    if external_level_secondary and (not (distributed_bank_avalanche_mode3_variant or
            combined_bedform_bank or
            (external_physical_toe and (bank_avalanche_mode2_variant or
                                        combined_bedform_mode2))) or
            int(data['secondary_flow_model']) not in (1,2)):
        raise ValueError('External level secondary requires an explicit recursive bank transaction')
    if (type(external_physical_toe) is not bool or
            (external_physical_toe and not
             (physical_toe_mode2_variant or bank_avalanche_mode2_variant or
              combined_bedform_mode2))):
        raise ValueError('External physical-toe ownership requires explicit mode-2 physics')
    if external_boundary_schedule:
        from .boundary_schedule import validate_tables
        if (scheduled or not (distributed_bank_avalanche_mode3_variant or
                combined_bedform_bank) or
                not np.isin(data['mode'], (2, 6)).all() or
                np.any(data.get('sampled', 0)) or np.any(data.get('section_control', 0)) or
                np.any(data.get('nseries', 0)) or np.any(data.get('nmorfac', 0)) or
                np.any(data.get('fraction_series_concentration', 0))):
            raise ValueError('External bank schedule requires unsampled stage/total-Q and class bed feed')
        validate_tables(data)
    if evolving_toe_variant and not (physical_toe_mode2_variant or
            bank_avalanche_mode2_variant or combined_bedform_mode2):
        raise ValueError('Evolving toe mapping requires the physical mode-2 transaction')
    seed_only = 'toe_geometry_seed_only' in data
    if seed_only:
        marker = np.asarray(data['toe_geometry_seed_only'])
        if (marker.shape != () or marker.dtype.kind not in 'iu' or int(marker) != 1
                or not evolving_toe_variant):
            raise ValueError('Geometry-only toe seeds require an explicit evolving GPU map')
    if 'viscosity_equilibrated_inlet' not in data:
        raise ValueError('Re-prepare input with the current initializer: missing native inlet-profile identity')
    profile=np.asarray(data['viscosity_equilibrated_inlet'])
    if profile.shape!=(len(data['mode']),) or np.any(profile):
        raise ValueError('GPU boundary port requires stock conveyance; viscosity-equilibrated inlets are not yet ported')
    validate_secondary_scope(data)
    viscosity=data.get('horizontal_viscosity',0.)
    if np.asarray(viscosity).size!=1 or not np.isfinite(viscosity) or float(viscosity)<0:
        raise ValueError('Horizontal viscosity must be finite and nonnegative')
    bedform_flag=np.asarray(data.get('bedform_roughness'))
    if bedform_flag.shape!=() or bedform_flag.dtype.kind not in 'iub' or int(bedform_flag) not in (0,1):
        raise ValueError('Explicit binary bedform-roughness control required')
    bedform_enabled=bool(int(bedform_flag))
    if combined_bedform_bank and not bedform_enabled:
        raise ValueError('Recursive bedform/bank composition requires bedform feedback ON')
    if g3_bank_secondary_variant and not bank_avalanche_mode2_variant:
        raise ValueError('G3 prognostic bank/avalanche scope requires the combined mode-2 transaction')
    if sum((bool(bedform_variant),bool(general_avalanche_variant),
            bool(physical_toe_mode2_variant),bool(bank_avalanche_mode2_variant),
            bool(distributed_bank_mode3_variant),
            bool(distributed_bank_avalanche_mode3_variant),
            combined_bedform_mode2,combined_bedform_mode3,
            bool(bedload_adaptation_variant)))>1:
        raise ValueError('GPU bedform, adaptation, general-avalanche and bank variants are exclusive')
    bedform_route = bool(bedform_variant or combined_bedform_bank)
    if bedform_enabled and not bedform_route:
        raise ValueError('Bedform roughness requires explicit GPU bedform_feedback_ssp33 variant')
    if bedform_route:
        if scheduled:
            raise ValueError('Bedform feedback and scheduled per-class inlet cannot be combined')
        modes = np.asarray(data['mode'])
        if combined_bedform_bank:
            if (modes.ndim != 1 or len(modes) < 1 or np.any(modes != 2) or
                    np.any(np.asarray(data['sampled'])) or
                    np.any(np.asarray(data['bedoffset'])) or
                    np.any(np.asarray(data['section_control']))):
                raise ValueError('Recursive bedform/bank composition requires constant stage boundaries')
        elif (not np.any(modes==1) or
              np.any(np.asarray(data['sampled'])) or np.any(np.asarray(data['bedoffset'])) or
              np.any(np.asarray(data['section_control']))):
            raise ValueError('Bedform variant requires an unsampled zero-offset mode-1 section inlet')
        if (int(data['bedlaw'])!=2 or int(data['mpm_friction_mode'])!=1 or
                int(data['bed_slope'])!=0 or int(data['hiding_exposure'])!=0 or
                int(data['secondary_flow_model'])!=0 or int(data['exchange_mode'])!=0 or
                int(data['nsubstrate_layers'])!=1 or float(data['morfac'])!=1. or
                bool(data.get('suspended_transport_active',False)) or
                np.any(data['suspended_mass']) or np.any(data['fraction_concentration']) or
                float(data.get('horizontal_viscosity',0.))!=0. or
                int(data.get('cpu_time_integrator',-1))!=33 or data['outlet_extension']!=0):
            raise ValueError('Bedform feedback variant requires stock unsuspended one-substrate MPM SSPRK33 scope')
        if bedform_enabled:
            from .bedform_roughness import validate_scope as validate_bedform_scope
            validate_bedform_scope(data,
                composed_bank_avalanche=combined_bedform_mode3,
                composed_physical_toe_avalanche=combined_bedform_mode2)
    if (general_avalanche_variant or bank_avalanche_mode2_variant or
            distributed_bank_avalanche_mode3_variant or combined_bedform_bank):
        if scheduled or (bedform_enabled and not combined_bedform_bank):
            raise ValueError('General-avalanche variant excludes scheduled inlet and bedform feedback')
        flag=np.asarray(data.get('general_avalanche'))
        if flag.shape!=() or flag.dtype.kind not in 'iub' or int(flag) not in (0,1):
            raise ValueError('Explicit binary general-avalanche control required')
        controls=('general_avalanche_wet_slope','general_avalanche_dry_slope',
                  'general_avalanche_time','general_avalanche_wet_depth_switch',
                  'general_avalanche_relaxation_length')
        values=[]
        for key in controls:
            if key not in data or np.asarray(data[key]).shape!=() or not np.isfinite(data[key]):
                raise ValueError('Missing/invalid prepared general-avalanche control '+key)
            values.append(float(data[key]))
        if min(values[0],values[1],values[2],values[4])<=0 or values[3]<0:
            raise ValueError('Invalid general-avalanche repose/diffusion controls')
        if general_avalanche_variant and (int(data['bedlaw'])!=0 or int(data['nsubstrate_layers'])!=1 or
                int(data['secondary_flow_model'])!=0 or int(data['exchange_mode'])!=0 or
                float(data['morfac'])!=1. or int(data.get('cpu_time_integrator',-1))!=33 or
                np.any(data['mode']) or len(data['mode'])!=0 or
                np.any(data['suspended_mass']) or np.any(data['fraction_concentration']) or
                bool(data.get('suspended_transport_active',False)) or
                float(data.get('horizontal_viscosity',0.))!=0. or data['outlet_extension']!=0):
            raise ValueError('General-avalanche variant requires closed, unsuspended bedlaw-0 SSPRK33 scope')
    if (physical_toe_mode2_variant or bank_avalanche_mode2_variant or
            combined_bedform_mode2):
        if scheduled or (bedform_enabled and not combined_bedform_mode2):
            raise ValueError('Physical-toe mode 2 excludes scheduled inlet and bedform feedback')
        mode=np.asarray(data.get('bank_toe_erosion_mode'))
        enabled=np.asarray(data.get('bank_failure'))
        if (mode.shape!=() or enabled.shape!=() or
                mode.dtype.kind not in 'iu' or enabled.dtype.kind not in 'biu' or
                int(mode)!=2 or int(enabled)!=1):
            raise ValueError('Physical-toe mode 2 requires native bank_failure=1 and bank_toe_erosion_mode=2')
        if 'bank_toe_capacity_fraction' not in data or 'bank_critical_undercut_m' not in data:
            raise ValueError('Missing physical-toe capacity/critical-undercut controls')
        capacity=np.asarray(data.get('bank_toe_capacity_fraction'))
        critical=np.asarray(data.get('bank_critical_undercut_m'))
        if (capacity.shape!=() or critical.shape!=() or
                capacity.dtype.kind not in 'if' or critical.dtype.kind not in 'if' or
                not np.isfinite(capacity) or not np.isfinite(critical) or
                not 0.<=float(capacity)<=1. or float(critical)<=0.):
            raise ValueError('Invalid physical-toe capacity/critical-undercut controls')
        secondary_model=int(data['secondary_flow_model'])
        allowed_secondary=(0,2) if g3_bank_secondary_variant else (0,)
        if (int(data['bedlaw'])!=2 or int(data['mpm_friction_mode'])!=1 or
                int(data['bed_slope'])!=0 or int(data['hiding_exposure'])!=0 or
                secondary_model not in allowed_secondary or int(data['exchange_mode'])!=0 or
                float(data['morfac'])!=1. or int(data.get('cpu_time_integrator',-1))!=33 or
                np.any(data['suspended_mass']) or np.any(data['fraction_concentration']) or
                bool(data.get('suspended_transport_active',False)) or
                float(data.get('horizontal_viscosity',0.))!=0. or
                data['outlet_extension']!=0):
            raise ValueError('Physical-toe mode 2 requires unsuspended fixed-grid MPM SSPRK33 scope')
        nc=np.asarray(data['q']).shape[1];nf=len(data['owner'])
        map_shapes={'bank_material_mask':(nc,), 'physical_toe_cell':(nc,),
                    'physical_toe_donor':(nc,), 'physical_toe_face':(nc,),
                    'physical_toe_distance':(nc,), 'physical_toe_status':(nc,),
                    'is_toe_cell':(nc,), 'is_toe_donor':(nc,),
                    'is_toe_face':(nf,), 'candidate_bank':(nf,),
                    'candidate_toe':(nf,)}
        for key,shape in map_shapes.items():
            if key not in data or np.asarray(data[key]).shape!=shape:
                raise ValueError('Missing/invalid prepared physical-toe map '+key)
        flags=('bank_material_mask','is_toe_cell','is_toe_donor','is_toe_face')
        for key in flags:
            value=np.asarray(data[key])
            if value.dtype.kind not in 'biu' or not np.isin(value,(0,1)).all():
                raise ValueError('Invalid physical-toe map flag '+key)
        ids=('physical_toe_cell','physical_toe_donor','physical_toe_face',
             'physical_toe_status','candidate_bank','candidate_toe')
        for key in ids:
            value=np.asarray(data[key])
            if value.dtype.kind not in 'iu':
                raise ValueError('Invalid physical-toe map index '+key)
        distances=np.asarray(data['physical_toe_distance'])
        if not np.isfinite(distances).all() or np.any(distances<0):
            raise ValueError('Invalid physical-toe map distances')
        map_status=np.asarray(data.get('toe_mapping_status'))
        if map_status.shape!=() or map_status.dtype.kind not in 'iu' or int(map_status)!=0:
            raise ValueError('Native physical-toe mapping did not succeed')
        material=np.asarray(data['bank_material_mask'],dtype=bool)
        per_cell=np.asarray(data['physical_toe_status'])
        if (not np.any(material) or not np.any(data['is_toe_face']) or
                not np.isin(per_cell[material], (1,2) if seed_only else (1,)).all() or
                np.any(per_cell[~material]!=0)):
            raise ValueError('Physical-toe map is empty, ambiguous or incomplete')
        for key,size in (('physical_toe_cell',nc),('physical_toe_donor',nc),('physical_toe_face',nf)):
            values=np.asarray(data[key])[material]
            if np.any(values<0) or np.any(values>=size):
                raise ValueError('Bank material has invalid mapped '+key)
    if type(experimental_morfac_mode3) is not bool:
        raise ValueError('Experimental mode-3 MorFac selector must be boolean')
    if experimental_morfac_mode3 and not (distributed_bank_mode3_variant or
            distributed_bank_avalanche_mode3_variant or combined_bedform_mode3):
        raise ValueError('Experimental MorFac is restricted to explicit mode 3')
    if (distributed_bank_mode3_variant or
            distributed_bank_avalanche_mode3_variant or combined_bedform_mode3):
        if scheduled or (bedform_enabled and not combined_bedform_mode3):
            raise ValueError('Distributed-bank mode 3 excludes scheduled inlet and bedform feedback')
        mode=np.asarray(data.get('bank_toe_erosion_mode'))
        enabled=np.asarray(data.get('bank_failure'))
        if (mode.shape!=() or enabled.shape!=() or
                mode.dtype.kind not in 'iu' or enabled.dtype.kind not in 'biu' or
                int(mode)!=3 or int(enabled)!=1):
            raise ValueError('Distributed-bank mode 3 requires native bank_failure=1 and bank_toe_erosion_mode=3')
        for key,lower,upper in (('bank_distributed_transport_fraction',0.,1.),
                                ('bank_distributed_critical_slope',0.,float('inf')),
                                ('bank_critical_undercut_m',0.,float('inf'))):
            value=np.asarray(data.get(key))
            if (value.shape!=() or value.dtype.kind not in 'if' or
                    not np.isfinite(value) or float(value)<lower or
                    float(value)>upper or (lower==0. and upper==float('inf') and float(value)==0.)):
                raise ValueError('Missing/invalid distributed-bank control '+key)
        slope_method=np.asarray(data.get(
            'bank_distributed_slope_method',
            'accepted_flux_distributed_slope'))
        if (slope_method.shape!=() or slope_method.dtype.kind not in 'USO' or
                str(slope_method.item()) not in (
                    'accepted_flux_distributed_slope',
                    'accepted_flux_published_net_gradient')):
            raise ValueError('Missing/invalid distributed-bank slope method')
        data['bank_distributed_slope_method']=str(slope_method.item())
        mask=np.asarray(data.get('bank_material_mask'))
        if (mask.shape!=(np.asarray(data['q']).shape[1],) or
                mask.dtype.kind not in 'biu' or not np.isin(mask,(0,1)).all() or
                not np.any(mask)):
            raise ValueError('Explicit nonempty native bank material mask required for mode 3')
        mode3_morfac=float(data['morfac'])
        morfac_ok=(mode3_morfac==1. or
                   (experimental_morfac_mode3 and 1.<mode3_morfac<=25. and
                    not np.any(data.get('nmorfac',0))))
        if (int(data['bedlaw'])!=2 or int(data['mpm_friction_mode'])!=1 or
                int(data['bed_slope'])!=0 or int(data['hiding_exposure'])!=0 or
                int(data['secondary_flow_model']) not in ((0,1,2) if external_level_secondary else (0,)) or int(data['exchange_mode'])!=0 or
                int(data.get('cpu_time_integrator',-1))!=33 or
                np.any(data['suspended_mass']) or np.any(data['fraction_concentration']) or
                bool(data.get('suspended_transport_active',False)) or
                float(data.get('horizontal_viscosity',0.))!=0. or
                not morfac_ok or data['outlet_extension']!=0):
            raise ValueError('Distributed-bank mode 3 requires fixed-grid unsuspended MPM SSPRK33 scope')
    if bedload_adaptation_variant:
        length=np.asarray(data.get('bedload_adaptation_length_diameters'))
        flag=np.asarray(data.get('bedload_adaptation'))
        if (flag.shape!=() or flag.dtype.kind not in 'biu' or int(flag)!=1 or
                length.shape!=() or length.dtype.kind!='f' or
                not np.isfinite(length) or float(length)<=0.):
            raise ValueError('Adaptation variant requires enabled native flag and exported positive length')
        modes=np.asarray(data['mode'])
        if (scheduled or bedform_enabled or int(data['bedlaw'])!=2 or
                int(data['mpm_friction_mode'])!=1 or int(data['bed_slope'])!=0 or
                int(data['secondary_flow_model'])!=0 or int(data['exchange_mode'])!=0 or
                int(data['nsubstrate_layers'])!=1 or float(data['morfac'])!=1. or
                int(data.get('cpu_time_integrator',-1))!=33 or
                data['outlet_extension']!=0 or
                len(modes)!=2 or np.count_nonzero(modes==6)!=1 or
                np.count_nonzero(modes==2)!=1 or
                np.any(data['suspended_mass']) or np.any(data['fraction_concentration']) or
                bool(data.get('suspended_transport_active',False)) or
                float(data.get('horizontal_viscosity',0.))!=0.):
            raise ValueError('Adaptation variant requires stock mode6/mode2 unsuspended fixed-grid MPM SSPRK33 scope')
    for key in ('amr_enabled','bedload_adaptation',
                'bank_failure','bank_avalanche','general_avalanche','bank_front_tracking','bank_erosion_fraction'):
        if key not in data:raise ValueError('Explicit physics control required: '+key)
        if np.any(data[key]) and not ((key=='bedload_adaptation' and bedload_adaptation_variant) or
                                     (key=='general_avalanche' and (general_avalanche_variant or
                                        bank_avalanche_mode2_variant or
                                        distributed_bank_avalanche_mode3_variant or
                                        combined_bedform_bank)) or
                                     (key=='bank_failure' and (physical_toe_mode2_variant or
                                                               bank_avalanche_mode2_variant or
                                                               combined_bedform_mode2 or
                                                               distributed_bank_mode3_variant or
                                                               distributed_bank_avalanche_mode3_variant or
                                                               combined_bedform_mode3))):
            raise ValueError('GPU solver does not yet support '+key)
    if 'exchange_mode' not in data:raise ValueError('Explicit physics control required: exchange_mode')
    exchange=np.asarray(data['exchange_mode'])
    if exchange.shape!=() or exchange.dtype.kind not in 'iu' or int(exchange) not in (0,1,2):
        raise ValueError('GPU fixed-grid suspension supports exchange modes 0, 1 and 2')
    if scheduled and int(exchange)!=0:
        # Only the all-wet, concentration-only, two-sand-class SSPRK33
        # engineering envelope has a native/GPU comparison and transactional
        # qualification. Keep every other scheduled/exchange combination
        # fail-closed rather than silently discarding an unported process.
        modes=np.asarray(data['mode'])
        forcing=np.asarray(data['forcing'])
        q=np.asarray(data['q'])
        layers=np.asarray(data['layer_mass'])
        feed_keys=('fraction_bed_input','bc_bed_input','fraction_series_bed_input')
        qualified=(
            int(exchange) in (1,2) and int(data['bedlaw'])==2 and
            int(data['scheme'])==6 and int(data.get('cpu_time_integrator',-1))==33 and
            int(data['nsubstrate_layers'])==1 and layers.ndim==3 and
            layers.shape[0:2]==(2,2) and q.ndim==2 and q.shape[0]==3 and
            np.all(q[0]>float(data['dry'])) and
            modes.shape==(2,) and np.array_equal(modes,[6,2]) and
            forcing.shape==(6,2) and forcing[0,0]>0 and
            np.isfinite(forcing[1,1]) and
            np.array_equal(np.asarray(data['nhydro_series']),[0,0]) and
            not np.any(data['nseries']) and
            np.array_equal(np.asarray(data['nfraction_series']),[[3,0],[3,0]]) and
            all(not np.any(data.get(key,0)) for key in feed_keys) and
            not np.any(forcing[3]) and
            np.array_equal(np.asarray(data['bed_boundary_mode']),[0,0]) and
            int(data['mpm_friction_mode'])==1 and int(data['bed_slope'])==0 and
            int(data['hiding_exposure'])==0 and
            int(data['secondary_flow_model'])==0 and
            int(data['sediment_roughness_mode'])==1 and
            np.array_equal(np.asarray(data['sediment_roughness_fraction']),[1,1]) and
            int(data['settling_mode'])==0 and
            np.asarray(data['ws_fraction']).shape==(2,) and
            np.all(np.asarray(data['ws_fraction'])>0) and
            float(data.get('horizontal_viscosity',0.))==0. and
            float(data['morfac'])==1. and data['outlet_extension']==0 and
            not np.any(data['sampled']) and not np.any(data['bedoffset']) and
            not np.any(data['section_control']) and
            not np.any(data['bedform_roughness']) and
            not np.any(data['general_avalanche']) and
            not np.any(data['bank_failure']) and
            not np.any(data['bedload_adaptation']) and
            not np.any(data['amr_enabled']))
        if not qualified:
            raise ValueError('Scheduled per-class inlet requires exchange_mode=0 outside the qualified all-wet two-class MPM exchange 1/2 scope')
    if int(exchange) in (1,2):
        settling=np.asarray(data.get('settling_mode'))
        if settling.shape!=() or settling.dtype.kind not in 'iu' or int(settling) not in (0,1):
            raise ValueError('GPU noncohesive exchange requires settling mode 0 or 1')
        kinds=np.asarray(data.get('sediment_type_fraction'))
        if kinds.shape!=(np.asarray(data['layer_mass']).shape[1],) or np.any(kinds!=0):
            raise ValueError('GPU noncohesive exchange requires sand fractions')
        hindered=np.asarray(data.get('hindered_settling_reference_concentration'))
        if (hindered.shape!=() or not np.isfinite(hindered).all() or
                float(hindered)<=0):
            raise ValueError('Explicit positive hindered-settling reference concentration required')
    if np.any(data.get('bank_toe_erosion_mode',0)) and not (
            physical_toe_mode2_variant or bank_avalanche_mode2_variant or
            distributed_bank_mode3_variant or
            distributed_bank_avalanche_mode3_variant or
            combined_bedform_bank):
        raise ValueError('GPU solver does not yet support bank-toe modes without explicit selector')
    if scheduled:
        from .scheduled_suspension_inlet import validate_input as validate_scheduled_input
        validate_scheduled_input(data)
    for key in ('nmorfac','nhydro_series','nseries','nfraction_series'):
        if external_boundary_schedule and key in ('nhydro_series','nfraction_series'):continue
        if scheduled and key in ('nhydro_series','nfraction_series'):continue
        if key in data and np.any(data[key]):raise ValueError('GPU solver does not yet support time series: '+key)
    law=np.asarray(data['bedlaw'])
    if law.shape!=() or law.dtype.kind not in 'iu' or int(law) not in (0,2,8,9) or int(np.asarray(data['scheme']).item())!=6:
        raise ValueError('GPU coupled solver requires scheme 6 and an explicit supported bed-load contract')
    if int(law)==0:
        if int(exchange) in (1,2) and float(data['morfac'])!=1.:
            raise ValueError('GPU no-bedload suspended exchange requires MorFac 1')
        if (int(np.asarray(data['mpm_friction_mode']).item())!=0 or
                int(np.asarray(data['bed_slope']).item())!=0 or
                int(np.asarray(data['hiding_exposure']).item())!=0 or
                int(np.asarray(data['secondary_flow_model']).item())!=0 or
                bool(data.get('morphological_acceleration',False))):
            raise ValueError('GPU no-bedload mode forbids MPM, slope, hiding, secondary and acceleration controls')
        for key in ('fraction_bed_input','bc_bed_input','series_bed_input','fraction_series_bed_input'):
            if key in data and np.any(np.asarray(data[key])!=0):
                raise ValueError('GPU no-bedload mode forbids bed feed: '+key)
        forcing=np.asarray(data.get('forcing'))
        bed_boundary=np.asarray(data.get('bed_boundary_mode'))
        if (forcing.ndim!=2 or forcing.shape[0]<4 or forcing.shape[1]!=len(data['mode']) or
                bed_boundary.shape!=(len(data['mode']),) or bed_boundary.dtype.kind not in 'iu'):
            raise ValueError('GPU no-bedload mode requires explicit shaped boundary controls')
        if np.any(forcing[3]!=0) or np.any(bed_boundary!=0):
            raise ValueError('GPU no-bedload mode forbids bed feed and equilibrium bed boundaries')
    q=np.asarray(data['q']);layers=np.asarray(data['layer_mass'])
    substrate=np.asarray(data.get('nsubstrate_layers',1))
    if substrate.shape!=() or substrate.dtype.kind not in 'iu' or not 1<=int(substrate)<=32:
        raise ValueError('GPU substrate-layer count must be 1 to 32')
    if (q.ndim!=2 or q.shape[0]!=3 or layers.ndim!=3 or
            layers.shape[0]!=1+int(substrate) or layers.shape[1]<1 or
            layers.shape[2]!=q.shape[1]):
        raise ValueError('Invalid coupled state dimensions')
    if not np.isfinite(q).all() or np.any(q[0]<0) or not np.isfinite(layers).all() or np.any(layers<0):
        raise ValueError('Invalid initial coupled state')
    if 'suspended_mass' not in data:raise ValueError('Explicit suspended inventory required')
    if 'fraction_concentration' not in data:
        raise ValueError('Explicit suspended boundary concentration required')
    suspended=np.asarray(data['suspended_mass']);concentration=np.asarray(data['fraction_concentration'])
    if (suspended.shape!=(layers.shape[1],q.shape[1]) or
            concentration.shape!=(layers.shape[1],len(data['mode'])) or
            not np.isfinite(suspended).all() or np.any(suspended<0) or
            not np.isfinite(concentration).all() or np.any(concentration<0)):
        raise ValueError('Invalid suspended inventory or boundary concentration')
    if np.asarray(data['rock']).shape!=(q.shape[1],) or not np.isfinite(data['rock']).all():
        raise ValueError('Invalid bedrock')
    if np.asarray(data['secondary_mass']).shape!=(q.shape[1],) or not np.isfinite(data['secondary_mass']).all():
        raise ValueError('Invalid secondary-flow inventory')
    if int(np.asarray(data['secondary_flow_model']).item())!=2 and np.any(data['secondary_mass']!=0):
        raise ValueError('OFF/equilibrium secondary flow has no transported hI inventory')
    if not np.isfinite(data['morfac']) or float(data['morfac'])<=0:raise ValueError('Invalid MorFac')
    if data['dt']<=0 or data['dry']<=0 or data['grav']<=0 or data['cfl_target']<=0:
        raise ValueError('Invalid numerical controls')


class CoupledFlux(ResidentFlux):
    """Same qualified hydraulic kernels with one pre-draining stress hook."""
    def step(self,secondary,predrain=None,postassemble=None):
        cp,a,b=self.cp,self.inputs,self.buffers
        nc,nf=np.int64(self.nc),np.int64(self.nf)
        dt,g,dry,target=[self.controls[k] for k in ('dt','grav','dry','cfl_target')]
        def launch(name,n,args):self.kernels[name](((n+127)//128,),(128,),args)
        launch('resident_prepare_faces',self.nf,(a['trace'],a['q'],a['bed'],a['normal'],a['distance'],
            a['owner'],a['neighbour'],a['kind'],nc,nf,g,dry,b['local'],b['closed']))
        launch('augmented_fwave_local',self.nf,(b['local'],nf,g,dry,b['riemann'],b['augmented_status']))
        launch('resident_assemble_flux',self.nf,(a['trace'],b['local'],b['riemann'],a['q'],a['bed'],a['normal'],
            a['owner'],a['neighbour'],a['kind'],a['boundary_mode'],a['sampled'],a['bedoffset'],a['hydro_constant'],
            b['closed'],nc,nf,g,dry,b['fl'],b['fr'],b['speed'],b['face_status']))
        if postassemble is not None:postassemble(b)
        stress=secondary.add_stress(a['trace'],a['kind'],b['closed'],b['fl'],b['fr'],b['speed'])
        if predrain is not None:predrain(b)
        cp.copyto(b['raw_fl'],b['fl']);cp.copyto(b['raw_fr'],b['fr'])
        launch('resident_draining',self.nc,(a['q'],a['area'],a['length'],b['fl'],a['offset'],a['faces'],
            a['signs'],a['active'],nc,nf,dt,b['theta']))
        launch('resident_limit_flux',self.nf,(a['q'],b['theta'],a['owner'],a['neighbour'],a['kind'],nc,nf,dry,
            b['fl'],b['fr'],b['face_status']))
        cp.max(a['q'][0],out=b['maxdepth'])
        launch('resident_gather_update',self.nc,(a['q'],a['area'],a['length'],a['normal'],b['fl'],b['fr'],
            b['speed'],b['maxdepth'],a['offset'],a['faces'],a['signs'],a['active'],nc,nf,dt,g,target,
            b['next'],b['cfl_ratio'],b['cell_status']))
        return b,stress['status']


class CoupledSolver:
    """Functional fixed-grid GPU solver; user-selected backend, no CPU fallback."""
    def __init__(self,data,*,source_folder=HERE,integrator='ssp43',scheduled=False,
                 bedform_variant=False,general_avalanche_variant=False,
                 physical_toe_mode2_variant=False,
                 bank_avalanche_mode2_variant=False,
                 g3_bank_secondary_variant=False,
                 distributed_bank_mode3_variant=False,
                 distributed_bank_avalanche_mode3_variant=False,
                 recursive_bedform_bank_avalanche_mode2_variant=False,
                 recursive_bedform_bank_avalanche_mode3_variant=False,
                 bedload_adaptation_variant=False, evolving_toe_variant=False,
                 distributed_flux_ownership='ssprk33', external_boundary_schedule=False,
                 external_level_secondary=False, external_live_sections=False,
                 external_physical_toe=False,
                 experimental_morfac_mode3=False):
        if (type(external_physical_toe) is not bool or
                (external_physical_toe and
                 (distributed_flux_ownership!='external_all_face' or integrator!='ssp33' or
                  not (physical_toe_mode2_variant or bank_avalanche_mode2_variant or
                       recursive_bedform_bank_avalanche_mode2_variant)))):
            raise ValueError('External physical-toe mode 2 requires explicit SSPRK33 all-face ownership')
        if (type(recursive_bedform_bank_avalanche_mode2_variant) is not bool or
                (recursive_bedform_bank_avalanche_mode2_variant and
                 (distributed_flux_ownership!='external_all_face' or
                  integrator!='ssp33' or not external_physical_toe))):
            raise ValueError('Recursive physical-toe bedform composition requires explicit SSPRK33 all-face ownership')
        if (type(recursive_bedform_bank_avalanche_mode3_variant) is not bool or
                (recursive_bedform_bank_avalanche_mode3_variant and
                 (distributed_flux_ownership!='external_all_face' or
                  integrator!='ssp33'))):
            raise ValueError('Recursive bedform/bank composition requires explicit SSPRK33 all-face ownership')
        if (type(external_live_sections) is not bool or (external_live_sections and
                (not external_boundary_schedule or not external_level_secondary or
                 distributed_flux_ownership!='external_all_face' or integrator!='ssp33' or
                 not (distributed_bank_avalanche_mode3_variant or
                      recursive_bedform_bank_avalanche_mode3_variant) or
                 any((scheduled,bedform_variant,general_avalanche_variant,physical_toe_mode2_variant,
                      bank_avalanche_mode2_variant,g3_bank_secondary_variant,
                      distributed_bank_mode3_variant,
                      recursive_bedform_bank_avalanche_mode2_variant,
                      recursive_bedform_bank_avalanche_mode3_variant,
                      bedload_adaptation_variant,evolving_toe_variant))))):
            raise ValueError('Surveyed recursive sections require explicit mode3 level-local ownership')
        if (type(external_level_secondary) is not bool or
                (external_level_secondary and
                 (distributed_flux_ownership!='external_all_face' or integrator!='ssp33'))):
            raise ValueError('External secondary requires explicit recursive SSPRK33 ownership')
        if (type(external_boundary_schedule) is not bool or
                (external_boundary_schedule and
                 (distributed_flux_ownership!='external_all_face' or integrator!='ssp33'))):
            raise ValueError('External boundary schedule requires explicit recursive SSPRK33 ownership')
        if (type(distributed_flux_ownership) is not str or
                distributed_flux_ownership not in ('ssprk33','external_all_face') or
                (distributed_flux_ownership != 'ssprk33' and not
                 (distributed_bank_mode3_variant or distributed_bank_avalanche_mode3_variant or
                  recursive_bedform_bank_avalanche_mode2_variant or
                  recursive_bedform_bank_avalanche_mode3_variant or
                  external_physical_toe))):
            raise ValueError('Explicit accepted-flux ownership is invalid')
        data=normalized(data)
        if external_live_sections:
            from braids_gpu.amr_boundary_forcing import validate_surveyed_scope
            validate_surveyed_scope(data)
        else:
            validate_scope(data,scheduled=scheduled,bedform_variant=bedform_variant,
                general_avalanche_variant=general_avalanche_variant,
                physical_toe_mode2_variant=physical_toe_mode2_variant,
                bank_avalanche_mode2_variant=bank_avalanche_mode2_variant,
                g3_bank_secondary_variant=g3_bank_secondary_variant,
                distributed_bank_mode3_variant=distributed_bank_mode3_variant,
                distributed_bank_avalanche_mode3_variant=distributed_bank_avalanche_mode3_variant,
                recursive_bedform_bank_avalanche_mode2_variant=
                    recursive_bedform_bank_avalanche_mode2_variant,
                recursive_bedform_bank_avalanche_mode3_variant=
                    recursive_bedform_bank_avalanche_mode3_variant,
                bedload_adaptation_variant=bedload_adaptation_variant,
                evolving_toe_variant=evolving_toe_variant,
                external_physical_toe=external_physical_toe,
                experimental_morfac_mode3=experimental_morfac_mode3,
                **({'external_level_secondary':True} if external_level_secondary else {}),
                **({'external_boundary_schedule':True} if external_boundary_schedule else {}))
        if integrator not in ('ssp43','ssp33'):
            raise ValueError('GPU coupled integrator must be explicitly ssp43 or ssp33')
        if scheduled and integrator!='ssp33':
            raise ValueError('Scheduled per-class inlet forcing requires SSPRK33')
        if (bedform_variant or recursive_bedform_bank_avalanche_mode2_variant or
                recursive_bedform_bank_avalanche_mode3_variant) and integrator!='ssp33':
            raise ValueError('Bedform feedback variant requires SSPRK33')
        if (general_avalanche_variant or bank_avalanche_mode2_variant or
                distributed_bank_avalanche_mode3_variant or
                recursive_bedform_bank_avalanche_mode2_variant or
                recursive_bedform_bank_avalanche_mode3_variant) and integrator!='ssp33':
            raise ValueError('General-avalanche variant requires SSPRK33')
        if (physical_toe_mode2_variant or
                recursive_bedform_bank_avalanche_mode2_variant) and integrator!='ssp33':
            raise ValueError('Physical-toe mode 2 requires SSPRK33')
        if (distributed_bank_mode3_variant or
                distributed_bank_avalanche_mode3_variant or
                recursive_bedform_bank_avalanche_mode3_variant) and integrator!='ssp33':
            raise ValueError('Distributed-bank mode 3 requires SSPRK33')
        if bedload_adaptation_variant and integrator!='ssp33':
            raise ValueError('Bedload adaptation requires SSPRK33')
        import cupy as cp
        from braids_core.windows_paths import filesystem_path
        self.cp=cp;self.data=data;self.folder=filesystem_path(source_folder);self.integrator=integrator
        self.recursive_bedform_bank_avalanche_mode2_variant = bool(
            recursive_bedform_bank_avalanche_mode2_variant)
        self.recursive_bedform_bank_avalanche_mode3_variant = bool(
            recursive_bedform_bank_avalanche_mode3_variant)
        self.bedform_variant=bool(bedform_variant or
            self.recursive_bedform_bank_avalanche_mode2_variant or
            self.recursive_bedform_bank_avalanche_mode3_variant)
        self.bedform_enabled=bool(data['bedform_roughness'])
        self.general_avalanche_variant=general_avalanche_variant
        self.bank_avalanche_mode2_variant=bool(bank_avalanche_mode2_variant or
            self.recursive_bedform_bank_avalanche_mode2_variant)
        self.g3_bank_secondary_variant=g3_bank_secondary_variant
        self.distributed_bank_avalanche_mode3_variant=bool(
            distributed_bank_avalanche_mode3_variant or
            self.recursive_bedform_bank_avalanche_mode3_variant)
        self.general_avalanche_enabled=(general_avalanche_variant or self.bank_avalanche_mode2_variant or
            self.distributed_bank_avalanche_mode3_variant) and bool(data['general_avalanche'])
        self.physical_toe_mode2_variant=bool(physical_toe_mode2_variant or
            self.bank_avalanche_mode2_variant)
        self.evolving_toe_variant=bool(evolving_toe_variant)
        self.evolving_toe=None
        self.toe_stage_index=0
        self.distributed_bank_mode3_variant=(distributed_bank_mode3_variant or
            self.distributed_bank_avalanche_mode3_variant)
        self.bedload_adaptation_variant=bedload_adaptation_variant
        self.distributed_stage_index=0
        self.distributed_flux_ownership=distributed_flux_ownership
        self.external_physical_toe=external_physical_toe
        self._distributed_external_clock=None
        self._distributed_external_phase=None
        self.avalanche_trial_substeps=[]
        self.inlet_schedule=None;self._inlet_schedule_stage=None;self._inlet_schedule_times=None
        self.external_boundary_schedule=external_boundary_schedule
        self.external_level_secondary=external_level_secondary
        self.external_live_sections=external_live_sections
        self.external_bedload_adaptation=external_live_sections and bool(data['bedload_adaptation'])
        self.live_components=None
        self.recursive_boundary_forcing=None
        self.nc=data['q'].shape[1];self.ng=data['layer_mass'].shape[1];self.nb=len(data['mode']);self.nf=len(data['owner'])
        self.bedload_active=int(data['bedlaw']) in (2,8,9)
        self.has_suspended=(int(data['exchange_mode'])!=0 or bool(data.get('suspended_transport_active',False)) or
            bool(np.any(data['suspended_mass'])) or bool(np.any(data['fraction_concentration'])) or
            bool(np.any(data.get('fraction_series_concentration',0))))
        self.state={'q':cp.asarray(data['q'],order='C',dtype=cp.float64),
                    'layers':cp.asarray(data['layer_mass'],order='C',dtype=cp.float64),
                    'secondary':cp.asarray(data['secondary_mass'],order='C',dtype=cp.float64)}
        self.ledger={'water':cp.zeros(self.nb),'bed':cp.zeros((self.ng,self.nb)),
                     'secondary':cp.zeros(self.nb),'source':cp.zeros(())}
        if self.has_suspended:
            self.state['suspended']=cp.asarray(data['suspended_mass'],order='C',dtype=cp.float64)
            self.ledger['suspended']=cp.zeros((self.ng,self.nb),dtype=cp.float64)
            self.ledger['exchange']=cp.zeros((self.ng,2),dtype=cp.float64)
        self.rock=cp.asarray(data['rock'],dtype=cp.float64)
        self.reconstruction=ResidentReconstruction(data,source_path=self.folder/'reconstruction.cu')
        self.flux=CoupledFlux(data,self.reconstruction,self.folder)
        # Viscosity is a separate supported post-step operator; this boundary
        # implementation accepts only stock conveyance, never equilibrated profiles.
        bcdata=boundary_data(data);bcdata['horizontal_viscosity']=0.
        if 'total_discharge_profile' in data and np.any(data['total_discharge_profile']):
            raise ValueError('GPU boundary port requires stock conveyance inlet profile')
        if external_live_sections:
            from braids_gpu.amr_boundary_forcing import surveyed_components
            with surveyed_components(self.folder,data) as components:
                self.live_components=components
                self.boundary=components.BoundaryControls(bcdata,self.flux.inputs,
                    source=self.folder/'boundaries.cu')
        elif external_boundary_schedule:
            self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu',
                allow_scheduled=True,allow_bedform_mode1=False)
        else:
            self.boundary=BoundaryControls(bcdata,self.flux.inputs,source=self.folder/'boundaries.cu',
                                       allow_scheduled=scheduled,
                                       allow_bedform_mode1=(self.bedform_variant and
                                           bool(np.any(np.asarray(data['mode']) == 1))))
        self.secondary=SecondaryTransport(data,self.reconstruction,source_path=self.folder/'secondary_transport.cu',
                                           curvature_source_path=self.folder/'secondary_curvature.cu')
        sediment_kwargs=dict(source_path=self.folder/'sediment_transport.cu',
                             augmented_source=self.folder/'augmented_fwave.cu')
        # Preserve the archived no-suspension constructor ABI used by the
        # independently qualified curved research adapter.
        if self.has_suspended:sediment_kwargs['external_suspension']=True
        if scheduled:sediment_kwargs['allow_scheduled']=True
        if external_boundary_schedule:sediment_kwargs['allow_scheduled']=True
        if self.bedform_variant:sediment_kwargs['allow_bedform']=True
        if (general_avalanche_variant or self.bank_avalanche_mode2_variant or
                self.distributed_bank_avalanche_mode3_variant):sediment_kwargs['allow_general_avalanche']=True
        if self.physical_toe_mode2_variant:sediment_kwargs['allow_physical_toe_mode2']=True
        if self.distributed_bank_mode3_variant:sediment_kwargs['allow_distributed_bank_mode3']=True
        if bedload_adaptation_variant or self.external_bedload_adaptation:
            sediment_kwargs['allow_adaptation']=True
        if external_live_sections:
            self.sediment=self.live_components.ResidentSedimentTransport(data,**sediment_kwargs)
        else:
            self.sediment=ResidentSedimentTransport(data,**sediment_kwargs)
        self.sediment.rotate_capacity=self.secondary.rotate_capacity
        if bedload_adaptation_variant or self.external_bedload_adaptation:
            from .bedload_adaptation import ResidentBedloadAdaptation
            self.sediment.adaptation=ResidentBedloadAdaptation(
                data,source_path=self.folder/'bedload_adaptation.cu')
            if self.external_bedload_adaptation:
                # Derived scratch is rebuilt on each topology, never remapped
                # as physical state. Initialize it before any trial snapshot.
                for value in self.sediment.adaptation.work.values():value.fill(0)
                self.sediment.adaptation.residual=0.
        self.bedform=None
        if self.bedform_enabled:
            from .bedform_roughness import ResidentBedformRoughness
            self.bedform=ResidentBedformRoughness(data,
                source_path=self.folder/'bedform_roughness.cu',
                composed_bank_avalanche=
                    self.recursive_bedform_bank_avalanche_mode3_variant,
                composed_physical_toe_avalanche=
                    self.recursive_bedform_bank_avalanche_mode2_variant)
            # Keep the bedform result and sediment input in independent device
            # allocations.  Recursive AMR transactions inventory every array
            # and deliberately reject aliases: a view into ``properties``
            # would make rollback order-dependent.  Each Euler stage copies
            # this derived row immediately after evaluating the bedform model.
            if self.sediment.geometry['manning_cell'].shape!=(self.nc,):
                raise ValueError('Invalid resident sediment Manning buffer')
        self.avalanche=None
        if self.general_avalanche_enabled:
            from .avalanche_operator import ResidentGeneralAvalanche
            from .avalanche_proposal import ResidentGeneralAvalancheProposal
            from .surface_transfer import ResidentSurfaceTransfer
            geometry=tuple(data[key] for key in ('area','owner','neighbour','offset','faces','signs'))
            proposal=ResidentGeneralAvalancheProposal(
                *geometry,*(data[key] for key in ('xy','mid','normal','length')),
                data['bulk_fraction'],dry_tolerance=float(data['dry_tolerance']),
                wet_slope=float(data['general_avalanche_wet_slope']),
                dry_slope=float(data['general_avalanche_dry_slope']),
                wet_depth_switch=float(data['general_avalanche_wet_depth_switch']),
                relaxation_length=float(data['general_avalanche_relaxation_length']),
                relaxation_time=float(data['general_avalanche_time']),
                source_path=self.folder/'avalanche_proposal.cu')
            transfer=ResidentSurfaceTransfer(*geometry[:6],nfrac=self.ng,
                source_path=self.folder/'surface_transfer.cu')
            self.avalanche=ResidentGeneralAvalanche(proposal,transfer,self.sediment,
                self.rock,nlayer=self.state['layers'].shape[0])
        self.suspended_faces=None;self.suspended_advection=None;self.suspended_exchange=None
        self.exchange_reduction=None;self.exchange_ledger_algorithm='none'
        if self.has_suspended:
            self.bc_concentration=cp.asarray(np.ascontiguousarray(data['fraction_concentration']),dtype=cp.float64)
            self.suspended_faces=ResidentSuspendedFaces(self.reconstruction,self.ng,self.nb,
                source_path=self.folder/'suspended_faces.cu')
            self.suspended_advection=ResidentSuspendedAdvection(data,self.ng,
                source_path=self.folder/'suspended_advection.cu')
            if int(data['exchange_mode']) in (1,2):
                self.suspended_exchange=ResidentNoncohesiveExchange(data,
                    source_path=self.folder/'suspended_exchange.cu')
                self.exchange_reduction=ResidentExchangeLedger(
                    self.suspended_advection.geometry['area'],self.ng,
                    source_path=self.folder/'exchange_ledger_parallel.cu')
                self.exchange_ledger_algorithm=self.exchange_reduction.method
                self.exchange_ledger=self.exchange_reduction.ledger
                self.exchange_ledger_status=self.exchange_reduction.status
                self.exchange_flag=cp.zeros((),dtype=cp.int32)
        self.exner=None;self.zero_bed_ledger=cp.zeros((self.ng,self.nb),dtype=cp.float64)
        if self.bedload_active:
            beddata=dict(data,mass=data['layer_mass'][0],raw=np.zeros((self.ng,self.nf)),
                         bulk=self.sediment.config['bulk'],nb=self.nb,active=np.ones(self.nc,dtype=np.int32))
            self.exner=ResidentBedUpdate(beddata,self.folder/'exner_update.cu')
            self.exner.geometry['active']=self.reconstruction.geometry['active']
        self.toe_supply=None;self.bank_undercut=None;self.toe_face_diagnostic=None
        self.distributed_bank=None
        self.bank_history={}
        self.toe_face_signed=None;self.toe_face_gross=None
        if self.physical_toe_mode2_variant:
            from .toe_supply import ResidentPhysicalToeSupply
            from .bank_undercut import ResidentPhysicalUndercut,FLOAT_HISTORY
            from .toe_face_diagnostic import ResidentToeFaceDiagnostic
            geometry=tuple(data[key] for key in ('area','owner','neighbour','offset','faces','signs'))
            self.toe_supply=ResidentPhysicalToeSupply(
                *geometry,data['length'],data['bulk_fraction'],
                data['is_toe_cell'],data['is_toe_face'],data['candidate_bank'],
                data['candidate_toe'],float(data['bank_toe_capacity_fraction']),
                self.state['layers'].shape[0],source_path=self.folder/'toe_supply.cu')
            self.bank_undercut=ResidentPhysicalUndercut(
                *geometry,data['length'],data['bulk_fraction'],
                data['is_toe_face'],data['candidate_bank'],data['candidate_toe'],
                data['is_toe_donor'],data['physical_toe_cell'],
                data['physical_toe_donor'],data['physical_toe_distance'],
                data['bank_material_mask'],self.state['layers'].shape[0],
                dry_tolerance=float(data['dry_tolerance']),
                critical_undercut=float(data['bank_critical_undercut_m']),
                source_path=self.folder/'bank_undercut.cu')
            self.toe_face_diagnostic=ResidentToeFaceDiagnostic(
                data['owner'],data['neighbour'],data['length'],
                data['is_toe_face'],data['candidate_bank'],data['candidate_toe'],
                self.ng,source_path=self.folder/'toe_face_diagnostic.cu',
                owner_orientation=self.evolving_toe_variant)
            self.bank_history={name:cp.zeros(self.nc,dtype=cp.float64)
                               for name in FLOAT_HISTORY}
            self.bank_history['collapse_count']=cp.zeros(self.nc,dtype=cp.int32)
            self.toe_face_signed=self.toe_face_diagnostic.signed
            self.toe_face_gross=self.toe_face_diagnostic.gross
            self.bank_flag=cp.zeros((),dtype=cp.int32)
            if self.evolving_toe_variant:
                from .evolving_toe_map import ResidentEvolvingToeMap
                self.evolving_toe=ResidentEvolvingToeMap(*geometry,
                    data['length'],data['xy'],data['mid'],data['bank_material_mask'],
                    data['bulk_fraction'],self.state['layers'].shape[0],
                    dry_tolerance=float(data['dry_tolerance']),
                    source_path=self.folder/'evolving_toe_map.cu')
                mapped=self.evolving_toe.build(self.bed(),self.state['q'][0],self.state['layers'])
                if int(mapped['status'].item()):
                    raise ValueError('Initial evolving toe mapping failed')
                self.evolving_toe.commit(mapped['status'])
                self.evolving_toe.bind(self.toe_supply,self.bank_undercut,self.toe_face_diagnostic)
                self.evolving_toe.register_private_history(self.bank_undercut.trial_history,
                                                          self.bank_history)
        if self.distributed_bank_mode3_variant:
            from .distributed_bank import ResidentDistributedBank,FLOAT_HISTORY
            geometry=tuple(data[key] for key in ('area','owner','neighbour','offset','faces','signs'))
            self.distributed_bank=ResidentDistributedBank(
                *geometry,*(data[key] for key in ('xy','normal','length','bulk_fraction',
                                                  'bank_material_mask')),
                self.state['layers'].shape[0],
                transport_fraction=float(data['bank_distributed_transport_fraction']),
                critical_slope=float(data['bank_distributed_critical_slope']),
                critical_undercut=float(data['bank_critical_undercut_m']),
                dry_tolerance=float(data['dry_tolerance']),
                slope_method=data.get('bank_distributed_slope_method',
                                      'accepted_flux_distributed_slope'),
                source_path=self.folder/'distributed_bank.cu',
                transfer_source=self.folder/'surface_transfer.cu')
            self.bank_history={name:cp.zeros(self.nc,dtype=cp.float64)
                               for name in FLOAT_HISTORY}
            self.bank_history['collapse_count']=cp.zeros(self.nc,dtype=cp.int32)
            self.bank_flag=cp.zeros((),dtype=cp.int32)
        self.euler_layers=cp.empty_like(self.state['layers']);self.euler_flag=cp.zeros((),dtype=cp.int32)
        self.euler_temp=cp.zeros((),dtype=cp.int32)
        stepper=CoupledRK43 if integrator=='ssp43' else CoupledRK33
        self.stepper=stepper(self.state,self.ledger,self.boundary.arrays['manning'],grav=data['grav'],dry=data['dry'],
            speed_limit=data['speed_limit'],chezy_minimum_depth=float(data.get('manning_friction_chezy_minimum_depth',0.)),
            source_folder=self.folder)
        self.diffusion=None
        if float(data.get('horizontal_viscosity',0.))>0:
            from .momentum_diffusion import ResidentMomentumDiffusion
            self.diffusion=ResidentMomentumDiffusion(data,source_path=self.folder/'momentum_diffusion.cu')
        if scheduled:
            from .scheduled_suspension_inlet import ResidentScheduledInletForcing
            self.inlet_schedule=ResidentScheduledInletForcing(data,source_path=self.folder/'boundary_schedule.cu')
        self.euler_calls=0

    def _merge(self,status,flag=None):
        if status is None:return
        if isinstance(status,(tuple,list)):
            for value in status:self._merge(value,flag)
            return
        if not status.size:return
        cp=self.cp
        cp.max(cp.where(status!=0,cp.int32(1),cp.int32(0)),out=self.euler_temp)
        self.stepper.hydro._launch('ti_merge_flag',1,(self.euler_temp,self.euler_flag if flag is None else flag))

    def bed(self,layers=None):
        return self.sediment.bed_elevation(self.state['layers'] if layers is None else layers,self.rock,
                                           self.reconstruction.geometry['bed'])

    def euler(self,state,dt,time):
        if self.distributed_flux_ownership=='external_all_face':
            clock=self._distributed_external_clock
            tolerance=64*np.finfo(float).eps*max(1.,abs(float(time)))
            if (self._distributed_external_phase!='open' or clock is None or
                    not np.isfinite([dt,time]).all() or not 0<dt<=clock[1] or
                    time<clock[0]-tolerance or time>clock[0]+clock[1]+tolerance):
                raise ValueError('External bank Euler is outside its open macro transaction')
        if self.external_boundary_schedule:
            raise ValueError('External boundary schedule requires the level-local Euler controller')
        if self.external_level_secondary:
            raise ValueError('External secondary flow requires the level-local Euler controller')
        cp=self.cp;self.euler_flag.fill(0)
        scheduled_record=None
        if self.inlet_schedule is not None:
            stage=self._inlet_schedule_stage
            if stage is None or stage>=3 or float(time)!=self._inlet_schedule_times[stage]:
                raise ValueError('Scheduled SSPRK33 Euler stage time/order changed')
            scheduled_record=self.inlet_schedule.records[stage]
            cp.copyto(self.boundary.arrays['forcing'],scheduled_record['hydraulic'])
            cp.copyto(self.sediment.geometry['bc_hydraulic'],scheduled_record['hydraulic'])
            cp.copyto(self.sediment.geometry['feed'],scheduled_record['feed'])
            if self.has_suspended:cp.copyto(self.bc_concentration,scheduled_record['concentration'])
            self._inlet_schedule_stage+=1
        bed=self.bed(state['layers']);q=state['q']
        bedform_status=None
        if self.bedform is not None:
            bedform_result=self.bedform.evaluate(q,state['layers'])
            bedform_status=bedform_result['status']
            cp.copyto(self.sediment.geometry['manning_cell'],
                      bedform_result['properties'][0])
        r=self.reconstruction.reconstruct(q)
        coupled_mpm=bool(getattr(self.sediment,'coupled_mpm',False))
        coupled_bedload=bool(getattr(self.sediment,'coupled_bedload',coupled_mpm))
        if coupled_bedload:
            self.sediment.prepare(q,bed,state['layers'],state['secondary'],r['active'])
        self.boundary.apply(q,r['trace'])
        stage_status=self.sediment.stage_characteristic(q,bed,r['face_value'],r['trace']) if coupled_mpm else None
        ss=self.secondary.reconstruct(q,bed,state['secondary'])
        self.flux.inputs['q']=q;self.flux.controls['dt']=np.float64(dt)
        sediment=None;coupled_face_status=None
        def predrain(buffers):
            nonlocal sediment,coupled_face_status
            face_input={'trace':r['trace'],'raw_fl':buffers['fl'],
                        'closed':buffers['closed'],'active':r['active']}
            sediment=self.sediment.fluxes(q,bed,state['layers'],state['secondary'],face_input,prepared=True)
            coupled_face_status=self.sediment.coupled_faces(
                {'local':buffers['local'],'riemann':buffers['riemann'],
                 'kind':self.flux.inputs['kind'],'closed':buffers['closed']},buffers)
        def postassemble(buffers):
            self.boundary.apply_pressure(q,bed,r['trace'],buffers['fl'],buffers['fr'],
                                         buffers['face_status'])
        f,stress_status=self.flux.step(self.secondary,
            predrain=predrain if coupled_bedload else None,postassemble=postassemble)
        if self.bedload_active and not coupled_bedload:
            face_input={'trace':r['trace'],'raw_fl':f['raw_fl'],'closed':f['closed'],'active':r['active']}
            sediment=self.sediment.fluxes(q,bed,state['layers'],state['secondary'],face_input)
        suspended_faces=None;suspended=None
        if self.has_suspended:
            suspended_faces=self.suspended_faces.step(q[0],state['suspended'],self.bc_concentration,r)
            suspended=self.suspended_advection.step(q[0],state['suspended'],f['next'][0],f['fl'][0],
                suspended_faces['left'],suspended_faces['right'],self.bc_concentration,r['hydro_constant'],dt)
        exner=None
        toe_diagnostic_status=None
        distributed_status=None
        cp.copyto(self.euler_layers,state['layers'])
        if self.bedload_active:
            exner=self.exner.step(state['layers'][0],sediment['raw'],dt,float(self.data['morfac']))
            cp.copyto(self.euler_layers[0],exner['next'])
            if self.physical_toe_mode2_variant and not self.external_physical_toe:
                toe_diagnostic_status=self.toe_face_diagnostic.capture(
                    exner['flux'],dt,self.toe_stage_index)
                self.toe_stage_index+=1
            if self.distributed_bank_mode3_variant and self.distributed_flux_ownership=='ssprk33':
                distributed_status=self.distributed_bank.capture(
                    exner['flux'],dt,self.distributed_stage_index)
                self.distributed_stage_index+=1
        avalanche=None
        if self.general_avalanche_variant:
            if self.avalanche is not None:
                avalanche=self.avalanche.step(self.euler_layers,f['next'][0],dt)
                cp.copyto(self.euler_layers,avalanche['layers'])
                self.avalanche_trial_substeps.append(int(avalanche['substeps']))
            else:self.avalanche_trial_substeps.append(0)
        rebalanced=self.sediment.rebalance(self.euler_layers)
        secondary=self.secondary.advect(q,f['next'],state['secondary'],f['fl'],dt)
        water=self.boundary.integrate_ledger(f['fl'],dt)
        self._merge((r['primitive_status'],r['reconstruction_status'],self.boundary.status,self.boundary.face_status,
            bedform_status,
            None if scheduled_record is None else scheduled_record['status'],
            f['augmented_status'],f['face_status'],f['cell_status'],ss['status'],stress_status,
            None if sediment is None else sediment['status'],
            None if exner is None else exner['status'],
            toe_diagnostic_status,
            distributed_status,
            None if avalanche is None else avalanche['status'],
            rebalanced,secondary['status'],stage_status,coupled_face_status,
            None if suspended_faces is None else suspended_faces['status'],
            None if suspended is None else suspended['status']))
        self.euler_calls+=1
        result_state={'q':f['next'],'layers':self.euler_layers,'secondary':secondary['next']}
        result_ledger={'water':water,'bed':self.zero_bed_ledger if exner is None else exner['ledger'],
                       'secondary':secondary['ledger']}
        if suspended is not None:
            result_state['suspended']=suspended['next_mass']
            result_ledger['suspended']=suspended['ledger']
        return dict(state=result_state,ledger=result_ledger,
            status=self.euler_flag,cfl=f['cfl_ratio'])

    def exchange(self,state,dt):
        if self.suspended_exchange is None:raise ValueError('No coupled suspended exchange operator')
        cp=self.cp;self.exchange_flag.fill(0)
        result=self.suspended_exchange.step(state['q'][0],state['q'][1:3],
            state['suspended'],state['layers'][0],dt)
        exchange_ledger=self.exchange_reduction.step(
            state['suspended'],state['layers'][0],result['suspended'],result['active'])
        self._merge((result['status'],exchange_ledger['status']),self.exchange_flag)
        cp.copyto(state['suspended'],result['suspended'])
        cp.copyto(state['layers'][0],result['active'])
        return exchange_ledger['ledger'],self.exchange_flag

    def relax(self,state,dt):
        result=self.secondary.relax(state['q'],state['secondary'],dt)
        return result['source_integral'],result['status']

    def post_source(self,state,dt):
        if self.diffusion is None:return None
        return self.diffusion.step(state['q'],self.bed(state['layers']),dt)

    def friction_roughness(self,state):
        if self.bedform is None:raise ValueError('No coupled equilibrium bedform roughness')
        result=self.bedform.evaluate(state['q'],state['layers'])
        return result['properties'][0],result['status']

    def physical_toe_pre_commit(self,state,dt):
        """Apply native mode-2 bank supply/collapse once to private RK trial.

        Histories and face diagnostics remain accepted-state data, not RK
        fields.  This hook runs after hydraulic/bed source and before the
        coupled step's final checks and status-gated global commit.
        """
        if not self.physical_toe_mode2_variant:
            raise ValueError('Physical-toe transaction requires explicit variant')
        if self.external_physical_toe:
            if (self._distributed_external_phase!='ready' or
                    self._distributed_external_clock is None or
                    self._distributed_external_clock[1]!=float(dt)):
                raise RuntimeError('Physical-toe macro step lacks its accepted face integral')
            self._distributed_external_phase='consumed'
        elif self.toe_stage_index!=3:
            raise RuntimeError('Physical-toe SSPRK33 omitted an Exner Euler stage')
        cp=self.cp
        self.bank_flag.fill(0)
        reference_bed=self.bed(self.state['layers'])
        toe=self.toe_supply.step(self.state['layers'],state['layers'],
            reference_bed=reference_bed,reference_depth=self.state['q'][0])
        collapse=self.bank_undercut.step(
            self.state['layers'],toe['trial_layers'],self.state['q'][0],
            reference_bed,toe['removed_mass'],self.bank_history,
            current_depth=state['q'][0])
        cp.copyto(state['layers'],collapse['trial_layers'])
        rebalanced=self.sediment.rebalance(state['layers'])
        self._merge((toe['status'],collapse['status'],rebalanced,
                     self.toe_face_diagnostic.validate_trial()),self.bank_flag)
        self.bank_trial=dict(toe=toe,collapse=collapse)
        if not self.bank_avalanche_mode2_variant:
            self._physical_mapping_trial(state)
        return self.bank_flag

    def _physical_mapping_trial(self,state):
        """Keep map_n for all RK legs; derive map_(n+1) after all bank sources."""
        if self.evolving_toe is None or int(self.bank_flag.item())!=0:
            return
        mapped=self.evolving_toe.build(self.bed(state['layers']),state['q'][0],state['layers'])
        self._merge(mapped['status'],self.bank_flag)
        if int(self.bank_flag.item())==0:
            self.evolving_toe.retire_changed_history(self.bank_undercut.trial_history)

    def physical_toe_commit(self,global_status):
        """Commit auxiliary state iff all coupled trial checks passed."""
        if not self.physical_toe_mode2_variant:
            raise ValueError('Physical-toe commit requires explicit variant')
        cp=self.cp
        for key,accepted in self.bank_history.items():
            trial=self.bank_undercut.trial_history[key]
            cp.copyto(accepted,cp.where(global_status==0,trial,accepted))
        self.toe_face_diagnostic.commit(global_status)
        if self.evolving_toe is not None:
            self.evolving_toe.commit(global_status)

    def bank_avalanche_pre_commit(self,state,dt):
        """Native combined ordering: post-RK bank failure, then one avalanche.

        Both operators write only the private trial. The stepper commits state,
        bank histories and face diagnostics together only after its global
        validity check, so a rejected trial cannot leak either process.
        """
        if not self.bank_avalanche_mode2_variant:
            raise ValueError('Bank/avalanche transaction requires explicit variant')
        flag=self.physical_toe_pre_commit(state,dt)
        if self.avalanche is None or int(flag.item())!=0:
            self.avalanche_trial_substeps.append(0)
            self._physical_mapping_trial(state)
            return flag
        result=self.avalanche.step(state['layers'],state['q'][0],dt)
        self.cp.copyto(state['layers'],result['layers'])
        self.avalanche_trial_substeps.append(int(result['substeps']))
        self._merge(result['status'],flag)
        # Avalanche may bury the last exposed block after bank collapse. Retire
        # only its geometric history on private state before the global commit.
        reconciled=self.bank_undercut.reconcile(state['layers'],state['q'][0],
            self.state['layers'],self.bed(self.state['layers']),
            self.bank_undercut.trial_history)
        self._merge(reconciled,flag)
        self._physical_mapping_trial(state)
        return flag

    def begin_external_bank_trial(self,dt,current):
        """Explicit recursive macro ownership; ordinary step() cannot use it."""
        if (self.distributed_flux_ownership!='external_all_face' or
                self._distributed_external_clock is not None or
                not np.isfinite([dt,current]).all() or dt<=0 or current<0):
            raise ValueError('Invalid/concurrent external distributed-bank trial')
        self._distributed_external_clock=(float(current),float(dt))
        self._distributed_external_phase='open'
        if self.external_physical_toe:
            self.toe_stage_index=0
            self.toe_face_diagnostic.begin_trial()
        else:
            self.distributed_bank.begin_trial()
        self.bank_trial=None
        self.distributed_stage_index=0
        self.avalanche_trial_substeps=[]

    def install_external_bank_integral(self,integral,*,dt,current):
        if (self.distributed_flux_ownership!='external_all_face' or
                self._distributed_external_phase!='open' or
                self._distributed_external_clock!=(float(current),float(dt))):
            raise ValueError('External bank integral is stale, repeated or outside its macro clock')
        if self.external_physical_toe:
            self.toe_face_diagnostic.set_accepted_face_integral(integral)
            # Preserve the public three-stage-complete invariant while making
            # the external register, rather than overwritten Euler buffers,
            # the sole diagnostic owner.
            self.toe_stage_index=3
        else:
            self.distributed_bank.set_accepted_face_integral(integral)
        self._distributed_external_phase='ready'

    def end_external_bank_trial(self):
        """Invalidate a consumed/rejected macro token; array rollback is caller-owned."""
        if self.distributed_flux_ownership!='external_all_face':
            raise ValueError('External bank trial belongs to another flux owner')
        self._distributed_external_clock=None
        self._distributed_external_phase=None
        if self.external_physical_toe:
            # A closed accepted/rejected macro has no pending three-stage
            # diagnostic. The completed root register has already been
            # committed or rolled back by its outer transaction.
            self.toe_stage_index=0

    def distributed_bank_pre_commit(self,state,dt):
        """Native-order mode-3 bank operator on the private accepted-step trial."""
        if not self.distributed_bank_mode3_variant:
            raise ValueError('Distributed-bank transaction requires explicit variant')
        if self.distributed_flux_ownership=='external_all_face':
            if (self._distributed_external_phase!='ready' or
                    self._distributed_external_clock is None or
                    self._distributed_external_clock[1]!=float(dt)):
                raise RuntimeError('Distributed-bank macro step lacks its final accepted face integral')
            self._distributed_external_phase='consumed'
        elif self.distributed_stage_index!=3:
            raise RuntimeError('Distributed-bank SSPRK33 omitted an Exner Euler stage')
        cp=self.cp
        self.bank_flag.fill(0)
        bed=self.bed(state['layers'])
        result=self.distributed_bank.step(state['layers'],state['q'][0],
                                          bed,self.bank_history,dt)
        cp.copyto(state['layers'],result['trial_layers'])
        rebalanced=self.sediment.rebalance(state['layers'])
        self._merge((result['status'],rebalanced),self.bank_flag)
        self.bank_trial=result
        return self.bank_flag

    def distributed_bank_avalanche_pre_commit(self,state,dt):
        """Native post-RK mode-3 bank failure, then one general avalanche.

        Both processes operate on the private SSPRK33 trial. A failed bank
        proposal skips avalanche; a failed avalanche prevents the enclosing
        state and all seven bank histories from committing.
        """
        if not self.distributed_bank_avalanche_mode3_variant:
            raise ValueError('Mode-3 bank/avalanche transaction requires explicit variant')
        flag=self.distributed_bank_pre_commit(state,dt)
        if self.avalanche is None or int(flag.item())!=0:
            self.avalanche_trial_substeps.append(0)
            return flag
        result=self.avalanche.step(state['layers'],state['q'][0],dt)
        self.cp.copyto(state['layers'],result['layers'])
        self.avalanche_trial_substeps.append(int(result['substeps']))
        self._merge(result['status'],flag)
        return flag

    def distributed_bank_commit(self,global_status):
        if not self.distributed_bank_mode3_variant:
            raise ValueError('Distributed-bank commit requires explicit variant')
        cp=self.cp
        for key,accepted in self.bank_history.items():
            trial=self.distributed_bank.trial_history[key]
            cp.copyto(accepted,cp.where(global_status==0,trial,accepted))

    def step(self,dt,time,*,commit=True):
        if self.distributed_flux_ownership!='ssprk33':
            raise ValueError('Externally owned bank flux requires the macro-step controller, not step()')
        if (self.general_avalanche_variant or self.bank_avalanche_mode2_variant or
                self.distributed_bank_avalanche_mode3_variant):
            self.avalanche_trial_substeps=[]
        if self.physical_toe_mode2_variant:
            self.toe_stage_index=0
            self.toe_face_diagnostic.begin_trial()
            self.bank_trial=None
        if self.distributed_bank_mode3_variant:
            self.distributed_stage_index=0
            self.distributed_bank.begin_trial()
            self.bank_trial=None
        if self.inlet_schedule is not None:
            if self._inlet_schedule_stage is not None:raise ValueError('Concurrent scheduled trials are unsupported')
            self.inlet_schedule.begin_trial(time,dt)
            self._inlet_schedule_times=(float(time),float(time+dt),float(time+.5*dt))
            self._inlet_schedule_stage=0
        try:
            result=self.stepper.step(dt,time,self.euler,self.sediment.rebalance,self.relax,
                                     exchange=self.exchange if self.suspended_exchange is not None else None,
                                     post_source=self.post_source,
                                     pre_commit=(self.bank_avalanche_pre_commit if self.bank_avalanche_mode2_variant else
                                                 self.distributed_bank_avalanche_pre_commit if self.distributed_bank_avalanche_mode3_variant else
                                                 self.physical_toe_pre_commit if self.physical_toe_mode2_variant else
                                                 self.distributed_bank_pre_commit if self.distributed_bank_mode3_variant else None),
                                     commit_aux=(self.physical_toe_commit if self.physical_toe_mode2_variant else
                                                 self.distributed_bank_commit if self.distributed_bank_mode3_variant else None),
                                     friction_roughness=self.friction_roughness if self.bedform is not None else None,
                                     commit=commit)
            if self.general_avalanche_variant:
                if len(self.avalanche_trial_substeps)!=3:
                    raise RuntimeError('General-avalanche SSPRK33 trial omitted an Euler stage')
                result['avalanche_substeps']=tuple(self.avalanche_trial_substeps)
            if self.bank_avalanche_mode2_variant:
                if len(self.avalanche_trial_substeps)!=1:
                    raise RuntimeError('Bank/avalanche SSPRK33 trial omitted post-RK repose')
                result['avalanche_substeps']=tuple(self.avalanche_trial_substeps)
            if self.distributed_bank_avalanche_mode3_variant:
                if len(self.avalanche_trial_substeps)!=1:
                    raise RuntimeError('Mode-3 bank/avalanche SSPRK33 trial omitted post-RK repose')
                result['avalanche_substeps']=tuple(self.avalanche_trial_substeps)
            if self.inlet_schedule is not None and self._inlet_schedule_stage!=3:
                raise RuntimeError('Scheduled SSPRK33 did not use all three Euler records')
            if self.physical_toe_mode2_variant and self.toe_stage_index!=3:
                raise RuntimeError('Physical-toe SSPRK33 did not capture all three Exner face fluxes')
            if self.distributed_bank_mode3_variant and self.distributed_stage_index!=3:
                raise RuntimeError('Distributed-bank SSPRK33 did not capture all three Exner face fluxes')
            return result
        finally:
            self._inlet_schedule_stage=None;self._inlet_schedule_times=None

    def fields(self):
        """Only scheduled output transfers arrays from device to host."""
        gaia={}
        if int(self.data['bedlaw'])==9 and not hasattr(self.sediment,'coupled_gaia'):
            raise ValueError('GAIA diagnostics require the corresponding sediment interface')
        if getattr(self.sediment,'coupled_gaia',False):
            # GAIA stresses are derived stage scratch, not persistent state.
            # Re-evaluate the accepted state at the output clock; the dedicated
            # kernel then checks its reconstructed capacity against the actual
            # production-capacity buffer before anything reaches the host.
            q=self.state['q'];bed=self.bed();reconstructed=self.reconstruction.reconstruct(q)
            rotation=self.sediment.prepare(q,bed,self.state['layers'],
                self.state['secondary'],reconstructed['active'])
            diagnostic=self.sediment.gaia_diagnostic(q,reconstructed['active'])
            statuses=(reconstructed['primitive_status'],reconstructed['reconstruction_status'],
                      self.sediment.buffers['cell_status'],rotation,diagnostic['status'])
            if any(bool(self.cp.any(item).item()) for item in statuses):
                raise RuntimeError('Invalid accepted GAIA secondary-stress diagnostic state')
            if (diagnostic['schema']!=self.sediment.gaia_diagnostic_schema or
                    tuple(diagnostic['fields'])!=self.sediment.gaia_diagnostic_fields):
                raise RuntimeError('GAIA secondary-stress diagnostic schema differs')
            gaia=dict(gaia_secondary_diagnostic_schema=np.int32(
                          self.sediment.gaia_diagnostic_schema),
                      gaia_secondary_diagnostic=self.cp.asnumpy(diagnostic['values']))
        adaptation={}
        if self.bedload_adaptation_variant:
            # Capacity is a derived stage-local solve, not restart state. At
            # output, recompute it from the accepted q/layers and unchanged
            # boundary feed so the diagnostic matches native output clocks.
            q=self.state['q'];bed=self.bed();reconstructed=self.reconstruction.reconstruct(q)
            # Equilibrium steering is derived, not transported. Its final
            # Euler-stage workspace need not equal the accepted RK state.
            equilibrium_status=None
            if self.secondary.model==1:
                equilibrium_status=self.secondary.reconstruct(q,bed,
                    self.state['secondary'])['status']
            rotation=self.sediment.prepare(q,bed,self.state['layers'],
                self.state['secondary'],reconstructed['active'])
            statuses=(reconstructed['primitive_status'],
                      reconstructed['reconstruction_status'],
                      self.sediment.buffers['cell_status'],rotation,equilibrium_status)
            if any(bool(self.cp.any(item).item()) for item in statuses if item is not None):
                raise RuntimeError('Invalid accepted bedload-adaptation diagnostic state')
            result=self.sediment.adaptation_result
            adaptation=dict(
                bedload_adaptation_equilibrium=self.cp.asnumpy(
                    self.sediment.adaptation.work['equilibrium']),
                bedload_adaptation_capacity=self.cp.asnumpy(
                    self.sediment.buffers['capacity']),
                bedload_adaptation_iterations=np.int32(result['iterations']),
                bedload_adaptation_residual=np.float64(result['residual']))
        bedform={}
        if self.bedform is not None:
            result=self.bedform.evaluate(self.state['q'],self.state['layers'])
            if bool(self.cp.any(result['status']).item()):
                raise RuntimeError('Invalid accepted bedform diagnostic state')
            properties=self.cp.asnumpy(result['properties'])
            bedform=dict(bedform_effective_manning=properties[0],
                bedform_dune_height=properties[1],bedform_dune_length=properties[2],
                bedform_total_ks=properties[5],
                bedform_iterations=self.cp.asnumpy(result['iterations']))
        return {**{k:self.cp.asnumpy(v) for k,v in self.state.items()},
                'bed_elevation':self.cp.asnumpy(self.bed()),
                **{'ledger_'+k:self.cp.asnumpy(v) for k,v in self.ledger.items()},
                **({'bank_history_'+k:self.cp.asnumpy(v) for k,v in self.bank_history.items()}
                   if self.physical_toe_mode2_variant or self.distributed_bank_mode3_variant else {}),
                **({'bank_toe_face_signed':self.cp.asnumpy(self.toe_face_signed),
                    'bank_toe_face_gross':self.cp.asnumpy(self.toe_face_gross)}
                   if self.physical_toe_mode2_variant else {}),
                **({'toe_map_'+k:self.cp.asnumpy(v)
                    for k,v in self.evolving_toe.persistent_arrays().items()}
                   if self.evolving_toe is not None else {}),
                 **bedform,**adaptation,**gaia}

    def gaia_fixed_inlet_diagnostic(self,current):
        """Evaluate the accepted state through the actual CUDA face pipeline.

        This is an endpoint-only validation diagnostic for the exact law-9
        envelope.  A positive one-ULP, non-committing Euler evaluation retains
        the production, post-limiter Exner face flux in its resident CUDA
        buffer; accepted state and ledgers are then verified byte-for-byte.
        """
        if (int(self.data['bedlaw'])!=9 or
                not getattr(self.boundary,'gaia_fixed',False)):
            raise ValueError('No explicit GAIA fixed inlet is active')
        if not np.isfinite(current) or float(current)<0:
            raise ValueError('Invalid GAIA endpoint diagnostic time')
        # The native CPU diagnostic uses dt=0 so its donor limiter tends to
        # the instantaneous face flux.  The production CUDA Exner operator
        # deliberately rejects nonpositive timesteps, so use one representable
        # time quantum at this clock.  This exercises the real limiter while
        # making its available-mass bound nonbinding for finite inventories.
        diagnostic_dt=max(float(np.spacing(max(1.,abs(float(current))))),
                          float(np.finfo(float).tiny))
        cp=self.cp
        accepted_before={key:value.copy() for key,value in self.state.items()}
        ledger_before={key:value.copy() for key,value in self.ledger.items()}
        stage_before={key:value.copy() for key,value in self.stepper.stage.items()}
        def same_bytes(left,right):
            return bool(cp.array_equal(left.reshape(-1).view(cp.uint8),
                                       right.reshape(-1).view(cp.uint8)).item())
        # The public transport boundary is deliberately bound once to the
        # resident SSPRK stage allocation.  Copy the accepted hydraulic state
        # into that scratch allocation; passing ``self.state['q']`` directly
        # would bypass the production storage-identity contract.
        calls=self.euler_calls
        try:
            for key,value in self.state.items():
                cp.copyto(self.stepper.stage[key],value)
            # Accepted state has already passed the production prepare/commit
            # checks.  Do not apply a future-step friction/source half-stage
            # here; the native CPU diagnostic likewise evaluates Euler directly.
            result=CoupledSolver.euler(self,self.stepper.stage,diagnostic_dt,float(current))
            if int(result['status'].item())!=0:
                raise RuntimeError('GAIA endpoint CUDA face evaluation failed')
            b=self.boundary.arrays
            fixed=cp.asnumpy(b['fixed_boundary_bed'])
            fixed_mode=cp.asnumpy(b['fixed_boundary_bed_mode'])
            offsets=cp.asnumpy(b['boundary_offset']);faces=cp.asnumpy(b['boundary_faces'])
            owner=self.flux.inputs['owner'];length=self.flux.inputs['length']
            bed=self.bed();offset=self.flux.inputs['bedoffset']
            # ``sediment.raw`` is the pre-Exner proposal. ``exner.flux`` is
            # the production MorFac-scaled, donor-limited flux. Capture it
            # before restoring the resident SSPRK scratch allocations.
            raw=self.exner.buffers['flux']
            rows=[]
            for boundary in np.flatnonzero(fixed_mode==1):
                selected=cp.asarray(faces[offsets[boundary]:offsets[boundary+1]],dtype=cp.int32)
                flux=float((-cp.sum(raw[:,selected]*length[selected])).item())
                if flux==0.:flux=0.
                cells=owner[selected]
                initial=fixed[cp.asnumpy(selected)]-cp.asnumpy(offset[selected])
                current_bed=cp.asnumpy(bed[cells])
                change=float(np.max(np.abs(current_bed-initial)))
                scale=max(1.,float(np.max(np.abs(initial))),float(np.max(np.abs(current_bed))))
                rows.append(dict(time_s=float(current),boundary_id=int(boundary)+1,
                    bedload_flux_kg_s=flux,owner_bed_max_abs_change_m=change,
                    owner_mobility_tolerance_m=64.*np.finfo(float).eps*scale,
                    owner_cells_mobile=bool(change>64.*np.finfo(float).eps*scale)))
            if not rows:
                raise RuntimeError('GAIA endpoint diagnostic found no fixed inlet')
            payload=dict(rows=rows,fixed_boundary_bed=np.ascontiguousarray(fixed,dtype='<f8'),
                         diagnostic_dt_s=diagnostic_dt)
        finally:
            self.euler_calls=calls
            for key,value in stage_before.items():
                cp.copyto(self.stepper.stage[key],value)
            accepted_unchanged=(
                all(same_bytes(value,self.state[key])
                    for key,value in accepted_before.items()) and
                all(same_bytes(value,self.ledger[key])
                    for key,value in ledger_before.items()))
            stage_restored=all(same_bytes(value,self.stepper.stage[key])
                               for key,value in stage_before.items())
            if not accepted_unchanged or not stage_restored:
                raise RuntimeError('GAIA endpoint diagnostic changed accepted or SSPRK state')
        return payload
