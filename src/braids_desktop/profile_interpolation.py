"""Display-only linear profiles; retain real gaps and the saved cell bounds."""
import numpy as np


def interpolate_intervals(distance,values,coordinates):
    distance=np.asarray(distance);values=np.asarray(values);coordinates=np.asarray(coordinates)
    if len(values)%2 or coordinates.shape!=(len(values),2):raise ValueError('Paired cell intervals required')
    out_d=[];out_v=[];out_xy=[];segment=[]
    def flush():
        if not segment:return
        first,last=segment[0],segment[-1]
        out_d.append(distance[2*first]);out_v.append(values[2*first]);out_xy.append(coordinates[2*first])
        for i in segment:
            out_d.append(.5*(distance[2*i]+distance[2*i+1]));out_v.append(values[2*i]);out_xy.append(.5*(coordinates[2*i]+coordinates[2*i+1]))
        out_d.append(distance[2*last+1]);out_v.append(values[2*last+1]);out_xy.append(coordinates[2*last+1])
        segment.clear()
    for i in range(len(values)//2):
        if np.isfinite(values[2*i:2*i+2]).all():segment.append(i)
        else:
            flush();out_d.extend(distance[2*i:2*i+2]);out_v.extend([np.nan,np.nan]);out_xy.extend(coordinates[2*i:2*i+2])
    flush()
    return np.asarray(out_d),np.asarray(out_v),np.asarray(out_xy)
