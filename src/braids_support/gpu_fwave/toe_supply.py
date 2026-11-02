"""Resident physical bank-toe supply with multilayer debit and face routing.

This applies mode 2 after an accepted Exner step, with a conservative current-
geometry limiter absent from the legacy native reference. It is deliberately
not passed through active-only ResidentSurfaceTransfer: native extraction
removes bank material from active then substrate layers; the routed face mass
is deposited at the recipient only. Passing it through a donor limiter again
would double-debit active mass and lose substrate composition.

The mapped physical toe geometry is supplied by the caller. This component
does not perform toe mapping, undercut accumulation, block collapse, avalanche
or AMR reflux, and a returned trial must be committed by the global solver.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .surface_transfer import validate_geometry

HERE=Path(__file__).resolve().parent
OPTIONS=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true','--ftz=false')


class ResidentPhysicalToeSupply:
    def __init__(self,area,owner,neighbour,offset,faces,signs,length,bulk,
                 is_toe_cell,is_toe_face,candidate_bank,candidate_toe,
                 capacity_fraction,nlayer,*,source_path=None):
        import cupy as cp
        topology,area=validate_geometry(area,owner,neighbour,offset,faces,signs)
        owner,neighbour,offset,faces,signs=topology
        nc=len(area);nf=len(owner)
        length=np.asarray(length,dtype=np.float64);bulk=np.asarray(bulk,dtype=np.float64)
        toe_cell=np.asarray(is_toe_cell);toe_face=np.asarray(is_toe_face)
        bank=np.asarray(candidate_bank);toe=np.asarray(candidate_toe)
        if (not isinstance(nlayer,(int,np.integer)) or not 1<=nlayer<=33 or
            bulk.ndim!=1 or not 1<=len(bulk)<=32 or not np.isfinite(bulk).all() or np.any(bulk<=0)):
            raise ValueError('Expected one to 33 total bed layers and positive grain bulk densities')
        if (length.shape!=(nf,) or not np.isfinite(length).all() or np.any(length<=0) or
            not np.isfinite(capacity_fraction) or not 0<=capacity_fraction<=1):
            raise ValueError('Invalid toe lengths or capacity fraction')
        if (toe_cell.shape!=(nc,) or toe_face.shape!=(nf,) or bank.shape!=(nf,) or toe.shape!=(nf,) or
            toe_cell.dtype.kind not in 'biu' or toe_face.dtype.kind not in 'biu' or
            bank.dtype.kind not in 'iu' or toe.dtype.kind not in 'iu' or
            np.any((toe_cell!=0)&(toe_cell!=1)) or np.any((toe_face!=0)&(toe_face!=1))):
            raise ValueError('Invalid physical toe mapping dimensions')
        bank=np.asarray(bank,dtype=np.int64);toe=np.asarray(toe,dtype=np.int64)
        toe_faces=[[] for _ in range(nc)];donor_faces=[[] for _ in range(nc)]
        for f in range(nf):
            if not toe_face[f]:
                if bank[f]!=-1 or toe[f]!=-1:
                    raise ValueError('Non-toe face has a candidate mapping')
                continue
            b=int(bank[f]);t=int(toe[f])
            if (neighbour[f]<0 or b<0 or b>=nc or t<0 or t>=nc or b==t or
                not toe_cell[t] or toe_cell[b] or
                not ((owner[f]==b and neighbour[f]==t) or
                     (owner[f]==t and neighbour[f]==b))):
                raise ValueError('Toe face is not its mapped physical material interface')
            toe_faces[t].append(f);donor_faces[b].append(f)
        def flatten(groups):
            counts=np.array([len(row) for row in groups],dtype=np.int32)
            return np.r_[0,np.cumsum(counts)].astype(np.int32),np.array(
                [f for row in groups for f in row],dtype=np.int32)
        toe_offset,toe_list=flatten(toe_faces)
        donor_offset,donor_list=flatten(donor_faces)
        source=Path(source_path or HERE/'toe_supply.cu').read_text()
        if '#define BRAIDS_TOE_INTERFACE_ABI 4' not in source:
            raise ValueError('Physical-toe Python/CUDA ABI mismatch: inventory-aware geometry kernels required')
        self.cp=cp;self.nc=nc;self.nf=nf;self.ng=len(bulk);self.nl=int(nlayer)
        self.capacity=np.float64(capacity_fraction)
        self.area=cp.asarray(area);self.bulk=cp.asarray(np.ascontiguousarray(bulk))
        self.owner,self.neighbour,self.offset,self.faces,self.signs=(cp.asarray(v) for v in topology)
        self.length=cp.asarray(np.ascontiguousarray(length))
        self.toe_cell=cp.asarray(np.ascontiguousarray(toe_cell,dtype=np.int32))
        self.toe_offset=cp.asarray(toe_offset);self.toe_faces=cp.asarray(toe_list)
        self.donor_offset=cp.asarray(donor_offset);self.donor_faces=cp.asarray(donor_list)
        self.candidate_bank=cp.asarray(np.ascontiguousarray(bank,dtype=np.int32))
        self.candidate_toe=cp.asarray(np.ascontiguousarray(toe,dtype=np.int32))
        names=('ts_demand','ts_face_requests','ts_donor_requests','ts_extract',
               'ts_route','ts_deposit','ts_budget','ts_geometry_state',
               'ts_geometry_scales','ts_limit_requests','ts_inventory_limit')
        module=cp.RawModule(code=source,options=OPTIONS,name_expressions=names)
        self.k={name:module.get_function(name) for name in names}
        self.trial_layers=cp.empty((self.nl,self.ng,nc),cp.float64)
        self.demand_mass=cp.empty((self.ng,nc),cp.float64)
        self.demand_volume=cp.empty(nc,cp.float64)
        self.face_request=cp.empty(nf,cp.float64)
        self.eligible_faces=cp.empty(nf,cp.int32)
        self.interface_area=cp.empty(nf,cp.float64)
        self.donor_request=cp.empty(nc,cp.float64)
        self.available_volume=cp.empty(nc,cp.float64)
        self.inventory_scale=cp.empty(nc,cp.float64)
        self.post_exner_bed=cp.empty(nc,cp.float64)
        self.requested_incoming=cp.empty(nc,cp.float64)
        self.donor_scale=cp.empty(nc,cp.float64);self.toe_scale=cp.empty(nc,cp.float64)
        self.removed_by_layer=cp.empty_like(self.trial_layers)
        self.removed_mass=cp.empty((self.ng,nc),cp.float64)
        self.actual_volume=cp.empty(nc,cp.float64)
        self.face_mass=cp.empty((self.ng,nf),cp.float64)
        self.recipient_gain=cp.empty((self.ng,nc),cp.float64)
        self.grain_budget=cp.empty((self.ng,2),cp.float64)
        self.failure=cp.zeros((),cp.int32);self._tmp=cp.zeros_like(self.failure)
        self.status={name:cp.empty(size,cp.int32) for name,size in
                     [('demand',nc),('face_request',nc),('donor_request',nc),
                      ('inventory_limit',nc),
                      ('geometry_state',nc),('geometry_scales',nc),
                      ('extract',nc),('route',nc),('deposit',nc),('budget',self.ng)]}
        self._mutable_names=('trial_layers','demand_mass','demand_volume',
            'face_request','eligible_faces','interface_area','donor_request',
            'available_volume','inventory_scale','post_exner_bed','requested_incoming',
            'donor_scale','toe_scale','removed_by_layer','removed_mass','actual_volume',
            'face_mass','recipient_gain','grain_budget','failure','_tmp')

    def _array(self,value,shape):
        if (not isinstance(value,self.cp.ndarray) or value.shape!=shape or
            value.dtype!=self.cp.float64 or not value.flags.c_contiguous):
            raise ValueError('Expected contiguous device FP64 array '+str(shape))

    def _launch(self,name,n,args):
        if n:self.k[name](((n+127)//128,),(128,),args)

    def _reject_private_overlap(self,inputs):
        # Pointer/range metadata only: do not synchronize or copy device data.
        # A contiguous view can start inside a private buffer, so pointer
        # equality is insufficient to preserve accepted input immutability.
        private=[getattr(self,name) for name in self._mutable_names]
        private.extend(self.status.values())
        ranges=[(value.data.ptr,value.data.ptr+value.nbytes) for value in private]
        for value in inputs:
            start=value.data.ptr;end=start+value.nbytes
            if any(start<stop and begin<end for begin,stop in ranges):
                raise ValueError('Input may not overlap private toe storage')

    def step(self,before_layers,accepted_exner_layers,*,reference_bed,reference_depth):
        """Return a private trial, using the undercut operator's pre-step geometry.

        Geometry is required: an old material mapping does not establish that
        the interface still exists after bed change or collapse deposition.
        No depth cutoff is added; any finite positive submerged height qualifies.
        """
        cp=self.cp;nc=np.int32(self.nc);nf=np.int32(self.nf)
        ng=np.int32(self.ng);nl=np.int32(self.nl)
        self._array(before_layers,(self.nl,self.ng,self.nc))
        self._array(accepted_exner_layers,(self.nl,self.ng,self.nc))
        self._array(reference_bed,(self.nc,))
        self._array(reference_depth,(self.nc,))
        self._reject_private_overlap((before_layers,accepted_exner_layers,
                                      reference_bed,reference_depth))
        self.failure.fill(0);self.face_request.fill(0);self.face_mass.fill(0)
        self.eligible_faces.fill(0);self.interface_area.fill(0.)
        self._launch('ts_demand',self.nc,(nc,ng,nl,before_layers,accepted_exner_layers,
            self.area,self.bulk,self.toe_cell,self.demand_mass,self.demand_volume,self.status['demand']))
        self._launch('ts_face_requests',self.nc,(nc,self.demand_volume,self.length,
            self.toe_offset,self.toe_faces,self.candidate_bank,reference_bed,reference_depth,
            self.capacity,self.face_request,self.eligible_faces,self.interface_area,
            self.status['face_request']))
        self._launch('ts_donor_requests',self.nc,(nc,self.donor_offset,self.donor_faces,
            self.face_request,self.donor_request,self.status['donor_request']))
        # Only physically available requests may contribute to another donor's
        # shared-recipient geometry bound. Keep each donor's allocated shares;
        # never reassign an exhausted donor's demand to a neighbouring bank.
        self._launch('ts_inventory_limit',self.nc,(nc,ng,nl,accepted_exner_layers,
            self.area,self.bulk,self.donor_offset,self.donor_faces,self.donor_request,
            self.face_request,self.available_volume,self.inventory_scale,
            self.status['inventory_limit']))
        self._launch('ts_donor_requests',self.nc,(nc,self.donor_offset,self.donor_faces,
            self.face_request,self.donor_request,self.status['donor_request']))
        self._launch('ts_geometry_state',self.nc,(nc,ng,nl,before_layers,
            accepted_exner_layers,self.bulk,reference_bed,self.toe_offset,
            self.toe_faces,self.face_request,self.post_exner_bed,
            self.requested_incoming,self.status['geometry_state']))
        self._launch('ts_geometry_scales',self.nc,(nc,self.area,self.length,
            self.eligible_faces,self.interface_area,self.donor_offset,self.donor_faces,
            self.toe_offset,self.toe_faces,self.candidate_bank,self.candidate_toe,
            self.post_exner_bed,self.donor_request,self.requested_incoming,
            self.donor_scale,self.toe_scale,self.status['geometry_scales']))
        self._launch('ts_limit_requests',self.nf,(nf,self.eligible_faces,
            self.candidate_bank,self.candidate_toe,self.donor_scale,self.toe_scale,
            self.face_request))
        self._launch('ts_donor_requests',self.nc,(nc,self.donor_offset,self.donor_faces,
            self.face_request,self.donor_request,self.status['donor_request']))
        self._launch('ts_extract',self.nc,(nc,ng,nl,accepted_exner_layers,self.area,self.bulk,
            self.donor_request,self.trial_layers,self.removed_by_layer,self.removed_mass,
            self.actual_volume,self.status['extract']))
        self._launch('ts_route',self.nc,(nc,nf,ng,self.donor_offset,self.donor_faces,
            self.face_request,self.donor_request,self.removed_mass,self.owner,self.candidate_bank,
            self.face_mass,self.status['route']))
        self._launch('ts_deposit',self.nc,(nc,nf,ng,self.area,self.offset,self.faces,
            self.signs,self.face_mass,self.trial_layers,self.recipient_gain,self.status['deposit']))
        self._launch('ts_budget',self.ng,(nc,ng,self.removed_mass,self.recipient_gain,
            self.grain_budget,self.status['budget']))
        for item in self.status.values():
            cp.max(cp.where(item!=0,cp.int32(1),cp.int32(0)),out=self._tmp)
            cp.maximum(self.failure,self._tmp,out=self.failure)
        return {'trial_layers':self.trial_layers,'demand_mass':self.demand_mass,
                'demand_volume':self.demand_volume,'face_request':self.face_request,
                'eligible_faces':self.eligible_faces,'interface_area':self.interface_area,
                'post_exner_bed':self.post_exner_bed,'requested_incoming':self.requested_incoming,
                'donor_scale':self.donor_scale,'toe_scale':self.toe_scale,
                'available_volume':self.available_volume,'inventory_scale':self.inventory_scale,
                'donor_request':self.donor_request,'removed_by_layer':self.removed_by_layer,
                'removed_mass':self.removed_mass,'actual_volume':self.actual_volume,
                'face_mass':self.face_mass,'recipient_gain':self.recipient_gain,
                'grain_budget':self.grain_budget,'status':self.failure,
                'detail_status':tuple(self.status.values())}
