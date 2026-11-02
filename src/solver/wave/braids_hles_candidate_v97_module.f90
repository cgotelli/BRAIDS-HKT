! Parallel candidate only: v90 owns accepted state, lag, receipts and clocks.
module braids_hles_candidate_v97_module
  use iso_c_binding, only: c_int, c_double
  use ieee_arithmetic, only: ieee_is_finite
  use omp_lib, only: omp_in_parallel
  implicit none
contains
  subroutine braids_hles_candidate_v97(nc,nf,nadj,threads,owner,neighbour,offset,faces,signs, &
      normal,length,distance,area,previous,mean_face,active_face,depth,chezy,active_cell, &
      coefficient,gamma,sigmat,flp,vicmol,dicmol,vonk,gravity,dry,hles_on,elder_on, &
      filtered,fluctuation,mean_cell,fluct_cell,complete,boundary,detvic_active,ptke, &
      hles,elder,dynamic_vic,dynamic_dic,status) bind(C)
    integer(c_int), value :: nc,nf,nadj,threads,hles_on,elder_on
    integer(c_int), intent(in) :: owner(nf),neighbour(nf),offset(nc+1),faces(nadj),signs(nadj)
    integer(c_int), intent(in) :: active_face(nf),active_cell(nc)
    real(c_double), intent(in) :: normal(nf,2),length(nf),distance(nf),area(nc)
    real(c_double), intent(in) :: previous(nf),mean_face(nf),depth(nc),chezy(nc)
    real(c_double), value :: coefficient,gamma,sigmat,flp,vicmol,dicmol,vonk,gravity,dry
    real(c_double), intent(out) :: filtered(nf),fluctuation(nf),mean_cell(nc,2),fluct_cell(nc,2)
    integer(c_int), intent(out) :: complete(nc),boundary(nc),detvic_active(nc),status
    real(c_double), intent(out) :: ptke(nc),hles(nc),elder(nc),dynamic_vic(nc),dynamic_dic(nc)
    integer :: face,cell,slot,other,count,bad
    real(c_double) :: axx,axy,ayy,bxm,bym,bxf,byf,nx,ny,w,det,scale
    real(c_double) :: dudx,dudy,dvdx,dvdy,uf,vf,inv_area,safe_h,safe_c,speed,b,ks,ustar
    logical :: valid
    bad=0
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) private(face)
    do face=1,nf
      filtered(face)=0.0_c_double
      if (coefficient>0.0_c_double .and. active_face(face)==1) &
        filtered(face)=coefficient*previous(face)+(1.0_c_double-coefficient)*mean_face(face)
      fluctuation(face)=mean_face(face)-filtered(face)
    end do
    !$omp end parallel do
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(cell,slot,face,count,axx,axy,ayy,bxm,bym,bxf,byf,nx,ny,w,det,scale) reduction(max:bad)
    do cell=1,nc
      axx=0.0_c_double; axy=0.0_c_double; ayy=0.0_c_double
      bxm=0.0_c_double; bym=0.0_c_double; bxf=0.0_c_double; byf=0.0_c_double; count=0
      do slot=offset(cell)+1,offset(cell+1)
        face=faces(slot)+1
        if (active_face(face)==0) cycle
        nx=normal(face,1); ny=normal(face,2); w=length(face)/distance(face)
        axx=axx+w*nx*nx; axy=axy+w*nx*ny; ayy=ayy+w*ny*ny
        bxm=bxm+w*nx*mean_face(face); bym=bym+w*ny*mean_face(face)
        bxf=bxf+w*nx*fluctuation(face); byf=byf+w*ny*fluctuation(face)
        count=count+1
      end do
      det=axx*ayy-axy*axy; scale=max(1.0_c_double,(axx+ayy)*(axx+ayy))
      mean_cell(cell,:)=0.0_c_double; fluct_cell(cell,:)=0.0_c_double; complete(cell)=0
      if (count>=2 .and. det>256.0_c_double*epsilon(det)*scale) then
        mean_cell(cell,1)=(ayy*bxm-axy*bym)/det; mean_cell(cell,2)=(axx*bym-axy*bxm)/det
        fluct_cell(cell,1)=(ayy*bxf-axy*byf)/det; fluct_cell(cell,2)=(axx*byf-axy*bxf)/det
        complete(cell)=1
      end if
      if (.not.all(ieee_is_finite(mean_cell(cell,:))) .or. &
          .not.all(ieee_is_finite(fluct_cell(cell,:)))) bad=1
    end do
    !$omp end parallel do
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(cell,slot,face,other,valid,nx,ny,uf,vf,dudx,dudy,dvdx,dvdy,inv_area) reduction(max:bad)
    do cell=1,nc
      dudx=0.0_c_double; dudy=0.0_c_double; dvdx=0.0_c_double; dvdy=0.0_c_double
      valid=active_cell(cell)==1 .and. complete(cell)==1
      boundary(cell)=0
      do slot=offset(cell)+1,offset(cell+1)
        face=faces(slot)+1
        other=owner(face)+1
        if (owner(face)==cell-1) other=neighbour(face)+1
        if (other<1) then
          boundary(cell)=1; valid=.false.; cycle
        end if
        if (complete(other)==0 .or. active_cell(other)==0) then
          valid=.false.; cycle
        end if
        nx=real(signs(slot),c_double)*normal(face,1); ny=real(signs(slot),c_double)*normal(face,2)
        uf=0.5_c_double*(fluct_cell(cell,1)+fluct_cell(other,1))
        vf=0.5_c_double*(fluct_cell(cell,2)+fluct_cell(other,2))
        dudx=dudx+uf*nx*length(face); dudy=dudy+uf*ny*length(face)
        dvdx=dvdx+vf*nx*length(face); dvdy=dvdy+vf*ny*length(face)
      end do
      ptke(cell)=0.0_c_double
      if (valid) then
        inv_area=1.0_c_double/area(cell)
        dudx=dudx*inv_area; dudy=dudy*inv_area; dvdx=dvdx*inv_area; dvdy=dvdy*inv_area
        ptke(cell)=2.0_c_double*(dudx*dudx+dvdy*dvdy)+(dudy+dvdx)**2
      end if
      if (.not.ieee_is_finite(ptke(cell))) bad=1
    end do
    !$omp end parallel do
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(cell,safe_h,safe_c,speed,b,ks,ustar) reduction(max:bad)
    do cell=1,nc
      detvic_active(cell)=0
      if (active_cell(cell)==1 .and. complete(cell)==1) detvic_active(cell)=1
      hles(cell)=0.0_c_double; elder(cell)=0.0_c_double
      dynamic_vic(cell)=0.0_c_double; dynamic_dic(cell)=0.0_c_double
      if (hles_on/=0 .and. detvic_active(cell)==1) then
        safe_h=max(dry,depth(cell)); safe_c=max(1.0_c_double,chezy(cell))
        speed=hypot(mean_cell(cell,1),mean_cell(cell,2))
        b=0.75_c_double*gravity*speed/(safe_h*safe_c*safe_c)
        ks=acos(-1.0_c_double)*flp/sqrt(area(cell))
        hles(cell)=(sqrt(gamma**2*sigmat**2*ptke(cell)+b*b)-b)/(ks*ks)
        if (elder_on/=0) then
          ustar=speed*sqrt(gravity)/safe_c
          elder(cell)=vonk*ustar*safe_h/6.0_c_double
        end if
        dynamic_vic(cell)=hles(cell)+elder(cell)+vicmol
        dynamic_dic(cell)=hles(cell)+elder(cell)+dicmol
      end if
      if (.not.ieee_is_finite(dynamic_vic(cell)) .or. .not.ieee_is_finite(dynamic_dic(cell))) bad=1
    end do
    !$omp end parallel do
    status=bad
  end subroutine
end module
