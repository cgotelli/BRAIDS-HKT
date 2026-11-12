"""Isolated Smart-Jaeggi bed-row consistency test, not a production method."""
from pathlib import Path
from validation.sediment.river_morphodynamics.sediment_wave import WaveEngine
from braids_core.sediment import class_wave_cache

INSERT='''  // Freeze the physical spiral angle, but retain its normal-vector direction
  // in the bed principal row. Smart-Jaeggi has an explicit -0.6*qb/h partial.
  if(law_params[0]==101){
   double alpha=.5*(cell[5*n+o]+cell[5*n+j]),norm=sqrt(1.+alpha*alpha);
   double directed=u+alpha*v;
   double radial=acceleration*derivative/h;
   der[0]=(-radial*directed-.6*acceleration*mag*directed/(h*sqrt(sp2)))/norm;
   der[1]=(factor+(radial-factor)*u*directed/sp2)/norm;
   der[2]=(alpha*factor+(radial-factor)*v*directed/sp2)/norm;
  }
'''


def install(a,folder):
    p=a.physics
    assert all(c.is_smart_jaggi for c in a.capacity)
    source=folder/'construction/runtime/owner_0001/split_wave.h'
    code=source.read_text();marker='  double b=-2.*u,c=u*u-gh*(1.+der[1]),d=-gh*(der[0]+v*der[2]);'
    assert code.count(marker)==1
    target=folder/'steered_wave.h';target.write_text(code.replace(marker,INSERT+marker))
    e=WaveEngine.__new__(WaveEngine)
    WaveEngine.__init__.__wrapped__(e,a.backend,p.flow.threads,source=target,names=class_wave_cache.NAMES)
    w=p.wave_engine;w.fn=e.fn;class_wave_cache.bind(w)
    if a.backend=='cpu':w.lib=e.lib;w.dirs=e.dirs
    else:w.module=e.module
    w._diagnostic_steered_wave=e
