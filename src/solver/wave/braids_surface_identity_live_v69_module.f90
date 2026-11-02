module braids_surface_identity_live_v69_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use omp_lib, only: omp_get_num_threads
    implicit none
    private
    public :: braids_surface_resolve_v69
    public :: braids_surface_after_exner_v69
    public :: braids_surface_depletion_project_v69
    public :: braids_surface_availability_v69
    public :: braids_surface_statistics_v69
contains

    subroutine braids_surface_resolve_v69(nfrac,ncell,mass,density, &
            retained,retained_supplied,identity,has_stock,valid,observed_threads)
        integer,intent(in) :: nfrac,ncell
        real(8),intent(in) :: mass(nfrac,ncell),density(nfrac)
        real(8),intent(in) :: retained(nfrac,ncell)
        logical,intent(in) :: retained_supplied
        real(8),intent(out) :: identity(nfrac,ncell)
        logical,intent(out) :: has_stock(ncell),valid
        integer,intent(out) :: observed_threads
        integer :: cell,fraction
        real(8) :: total,tol
        logical :: local_valid

        identity=0.d0;has_stock=.false.;valid=.true.;observed_threads=1
        if(nfrac<=0.or.ncell<=0.or.any(.not.ieee_is_finite(mass)).or. &
                any(mass<0.d0).or.any(.not.ieee_is_finite(density)).or. &
                any(density<=0.d0)) then
            valid=.false.;return
        endif
        tol=512.d0*epsilon(1.d0)*real(nfrac,8)
        !$omp parallel default(shared) private(cell,fraction,total,local_valid)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do schedule(static)
        do cell=1,ncell
            total=0.d0
            do fraction=1,nfrac
                total=total+mass(fraction,cell)/density(fraction)
            enddo
            if(total>0.d0) then
                has_stock(cell)=.true.
                do fraction=1,nfrac
                    identity(fraction,cell)=mass(fraction,cell)/ &
                                              density(fraction)/total
                enddo
            else
                local_valid=retained_supplied.and. &
                    all(ieee_is_finite(retained(:,cell))).and. &
                    all(retained(:,cell)>=0.d0).and. &
                    abs(sum(retained(:,cell))-1.d0)<=tol
                if(local_valid) then
                    identity(:,cell)=retained(:,cell)
                else
                    !$omp critical
                    valid=.false.
                    !$omp end critical
                endif
            endif
        enddo
        !$omp end do
        !$omp end parallel
    end subroutine braids_surface_resolve_v69

    subroutine braids_surface_after_exner_v69(nfrac,ncell,candidate_mass, &
            density,parent_identity,candidate_identity,valid,observed_threads)
        integer,intent(in) :: nfrac,ncell
        real(8),intent(in) :: candidate_mass(nfrac,ncell),density(nfrac)
        real(8),intent(in) :: parent_identity(nfrac,ncell)
        real(8),intent(out) :: candidate_identity(nfrac,ncell)
        logical,intent(out) :: valid
        integer,intent(out) :: observed_threads
        logical :: stock(ncell)
        call braids_surface_resolve_v69(nfrac,ncell,candidate_mass,density, &
            parent_identity,.true.,candidate_identity,stock,valid,observed_threads)
    end subroutine braids_surface_after_exner_v69

    subroutine braids_surface_depletion_project_v69(nfrac,ncell,parent_mass, &
            raw_candidate,raw_delta,cell_area,candidate,accepted_delta, &
            roundoff_transfer,depleted,valid,observed_threads)
        integer,intent(in) :: nfrac,ncell
        real(8),intent(in) :: parent_mass(nfrac,ncell)
        real(8),intent(in) :: raw_candidate(nfrac,ncell)
        real(8),intent(in) :: raw_delta(nfrac,ncell),cell_area(ncell)
        real(8),intent(out) :: candidate(nfrac,ncell)
        real(8),intent(out) :: accepted_delta(nfrac,ncell)
        real(8),intent(out) :: roundoff_transfer(nfrac)
        logical,intent(out) :: depleted(ncell),valid
        integer,intent(out) :: observed_threads
        real(8),allocatable :: removed(:,:)
        real(8) :: tolerance(nfrac),cell_tolerance,consistency
        integer :: cell,fraction
        logical :: local_valid

        candidate=0.d0;accepted_delta=0.d0;roundoff_transfer=0.d0
        depleted=.false.;valid=.true.;observed_threads=1
        if(nfrac<=0.or.ncell<=0.or.any(.not.ieee_is_finite(parent_mass)).or. &
           any(parent_mass<0.d0).or.any(.not.ieee_is_finite(raw_candidate)).or. &
           any(.not.ieee_is_finite(raw_delta)).or. &
           any(.not.ieee_is_finite(cell_area)).or.any(cell_area<=0.d0)) then
            valid=.false.;return
        endif
        allocate(removed(nfrac,ncell));removed=0.d0
        !$omp parallel default(shared) private(cell,fraction,tolerance, &
        !$omp& cell_tolerance,consistency,local_valid)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do schedule(static)
        do cell=1,ncell
            local_valid=.true.
            do fraction=1,nfrac
                tolerance(fraction)=2048.d0*epsilon(1.d0)* &
                    max(1.d0,abs(parent_mass(fraction,cell)))
                consistency=abs(raw_candidate(fraction,cell)- &
                    (parent_mass(fraction,cell)+raw_delta(fraction,cell)))
                if(raw_candidate(fraction,cell)<-tolerance(fraction).or. &
                   consistency>tolerance(fraction)) local_valid=.false.
                candidate(fraction,cell)=max(raw_candidate(fraction,cell),0.d0)
                accepted_delta(fraction,cell)=raw_delta(fraction,cell)
            enddo
            cell_tolerance=sum(tolerance)
            if(local_valid.and.sum(candidate(:,cell))<=cell_tolerance) then
                depleted(cell)=.true.
                removed(:,cell)=candidate(:,cell)*cell_area(cell)
                candidate(:,cell)=0.d0
                accepted_delta(:,cell)=-parent_mass(:,cell)
            endif
            if(.not.local_valid) then
                !$omp critical
                valid=.false.
                !$omp end critical
            endif
        enddo
        !$omp end do
        !$omp end parallel
        do fraction=1,nfrac
            roundoff_transfer(fraction)=sum(removed(fraction,:))
        enddo
        deallocate(removed)
    end subroutine braids_surface_depletion_project_v69

    subroutine braids_surface_availability_v69(nfrac,ncell,mass,identity, &
            outgoing_demand,fill,availability,valid,observed_threads)
        integer,intent(in) :: nfrac,ncell
        real(8),intent(in) :: mass(nfrac,ncell),identity(nfrac,ncell)
        real(8),intent(in) :: outgoing_demand(nfrac,ncell)
        real(8),intent(out) :: fill(nfrac,ncell),availability(nfrac,ncell)
        logical,intent(out) :: valid
        integer,intent(out) :: observed_threads
        integer :: cell,fraction
        fill=0.d0;availability=0.d0;valid=.true.;observed_threads=1
        if(any(.not.ieee_is_finite(mass)).or.any(mass<0.d0).or. &
           any(.not.ieee_is_finite(identity)).or.any(identity<0.d0).or. &
           any(.not.ieee_is_finite(outgoing_demand)).or. &
           any(outgoing_demand<0.d0)) then
            valid=.false.;return
        endif
        !$omp parallel default(shared) private(cell,fraction)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do collapse(2) schedule(static,256)
        do cell=1,ncell
            do fraction=1,nfrac
                if(mass(fraction,cell)<=0.d0) then
                    fill(fraction,cell)=0.d0
                else if(outgoing_demand(fraction,cell)>0.d0) then
                    fill(fraction,cell)=min(1.d0,mass(fraction,cell)/ &
                                                  outgoing_demand(fraction,cell))
                else
                    fill(fraction,cell)=1.d0
                endif
                availability(fraction,cell)=identity(fraction,cell)* &
                                             fill(fraction,cell)
            enddo
        enddo
        !$omp end do
        !$omp end parallel
    end subroutine braids_surface_availability_v69

    subroutine braids_surface_statistics_v69(nfrac,ncell,maxpoint,fraction, &
            mean_diameter,median_diameter,sediment_type,max_mud_type, &
            min_dxx_type,count,probability,log_diameter,dm,dg,d10,d50,d90, &
            dgsd,sand,mud,has_nonmud,valid,observed_threads)
        integer,intent(in) :: nfrac,ncell,maxpoint,max_mud_type,min_dxx_type
        integer,intent(in) :: sediment_type(nfrac),count(nfrac)
        real(8),intent(in) :: fraction(nfrac,ncell),mean_diameter(nfrac)
        real(8),intent(in) :: median_diameter(nfrac)
        real(8),intent(in) :: probability(maxpoint,nfrac)
        real(8),intent(in) :: log_diameter(maxpoint,nfrac)
        real(8),intent(out) :: dm(ncell),dg(ncell),d10(ncell),d50(ncell),d90(ncell)
        real(8),intent(out) :: dgsd(ncell),sand(ncell),mud(ncell)
        logical,intent(out) :: has_nonmud(ncell),valid
        integer,intent(out) :: observed_threads
        integer :: cell,l
        real(8) :: fracdiam,fracnonmud,weight,logdg,requested(3),dxx(3),tol
        logical :: point_valid

        dm=0.d0;dg=0.d0;d10=0.d0;d50=0.d0;d90=0.d0;dgsd=0.d0
        sand=0.d0;mud=0.d0;has_nonmud=.false.;valid=.true.;observed_threads=1
        requested=[.1d0,.5d0,.9d0]
        if(nfrac<=0.or.ncell<=0.or.maxpoint<=0.or. &
           any(.not.ieee_is_finite(fraction)).or.any(fraction<0.d0)) then
            valid=.false.;return
        endif
        tol=512.d0*epsilon(1.d0)*real(nfrac,8)
        !$omp parallel default(shared) private(cell,l,fracdiam,fracnonmud, &
        !$omp& weight,logdg,dxx,point_valid)
        !$omp single
        observed_threads=omp_get_num_threads()
        !$omp end single
        !$omp do schedule(static)
        do cell=1,ncell
            point_valid=abs(sum(fraction(:,cell))-1.d0)<=tol
            fracdiam=sum(fraction(:,cell),mask=sediment_type>=min_dxx_type)
            fracnonmud=sum(fraction(:,cell),mask=sediment_type>max_mud_type)
            mud(cell)=sum(fraction(:,cell),mask=sediment_type<=max_mud_type)
            has_nonmud(cell)=fracnonmud>0.d0
            if(fracdiam>0.d0) then
                logdg=0.d0
                do l=1,nfrac
                    if(sediment_type(l)>=min_dxx_type) then
                        weight=fraction(l,cell)/fracdiam
                        dm(cell)=dm(cell)+weight*mean_diameter(l)
                        logdg=logdg+weight*log(median_diameter(l))
                    endif
                enddo
                dg(cell)=exp(logdg)
                do l=1,nfrac
                    if(sediment_type(l)>=min_dxx_type.and. &
                       fraction(l,cell)>0.d0) then
                        weight=fraction(l,cell)/fracdiam
                        dgsd(cell)=dgsd(cell)+weight* &
                            (log(median_diameter(l))-logdg)**2
                    endif
                enddo
                dgsd(cell)=exp(sqrt(dgsd(cell)))
            endif
            if(fracnonmud>0.d0) sand(cell)=sum(fraction(:,cell), &
                mask=sediment_type>max_mud_type.and. &
                     mean_diameter<2.d-3)/fracnonmud
            call percentile_scan(requested,nfrac,maxpoint,fraction(:,cell), &
                sediment_type,min_dxx_type,count,probability,log_diameter, &
                dxx,point_valid)
            d10(cell)=dxx(1);d50(cell)=dxx(2);d90(cell)=dxx(3)
            if(.not.point_valid) then
                !$omp critical
                valid=.false.
                !$omp end critical
            endif
        enddo
        !$omp end do
        !$omp end parallel
    end subroutine braids_surface_statistics_v69

    subroutine percentile_scan(requested,nfrac,maxpoint,fraction,sediment_type, &
            min_dxx_type,count,probability,log_table,dxx,valid)
        integer,intent(in) :: nfrac,maxpoint,min_dxx_type
        integer,intent(in) :: sediment_type(nfrac),count(nfrac)
        real(8),intent(in) :: requested(3),fraction(nfrac)
        real(8),intent(in) :: probability(maxpoint,nfrac),log_table(maxpoint,nfrac)
        real(8),intent(out) :: dxx(3)
        logical,intent(inout) :: valid
        integer :: stage(nfrac),i,l,s,trigger,iterations,max_iterations
        real(8) :: fracdiam,fracfac,density,logdiam,logprevious
        real(8) :: accumulated,interval,required
        dxx=0.d0;stage=0;i=1;logprevious=0.d0;accumulated=0.d0
        required=requested(i);iterations=0
        fracdiam=sum(fraction,mask=sediment_type>=min_dxx_type)
        max_iterations=sum(max(count,0))+nfrac+12
        do while(required>accumulated)
            iterations=iterations+1
            if(iterations>max_iterations) then
                valid=.false.;return
            endif
            logdiam=999.999d0;trigger=0;density=0.d0
            do l=1,nfrac
                s=stage(l)
                if(s<count(l)) then
                    if(s>0) then
                        if(fracdiam>0.d0) then
                            fracfac=fraction(l)/fracdiam
                        else
                            fracfac=1.d0
                        endif
                        density=density+fracfac*(probability(s+1,l)- &
                            probability(s,l))/(log_table(s+1,l)-log_table(s,l))
                    endif
                    if(log_table(s+1,l)<logdiam) then
                        logdiam=log_table(s+1,l);trigger=l
                    endif
                endif
            enddo
            if(trigger==0) then
                dxx(i:)=exp(logprevious);return
            endif
            interval=density*(logdiam-logprevious)
            do while(interval>=required-accumulated)
                if(interval>0.d0) logprevious=logprevious+ &
                    (logdiam-logprevious)*(required-accumulated)/interval
                accumulated=required;dxx(i)=exp(logprevious)
                if(i==3) return
                i=i+1;required=requested(i)
                interval=density*(logdiam-logprevious)
            enddo
            accumulated=accumulated+interval;logprevious=logdiam
            if(stage(trigger)==0) then
                if(log_table(count(trigger),trigger)<=log_table(1,trigger)) then
                    stage(trigger)=max(5,count(trigger))
                    ! The locked compdiam source has a strict > here, which
                    ! leaves Dxx unassigned at an exact cumulative edge. Use the
                    ! right-continuous inverse CDF so the point owns the edge.
                    do while(fraction(trigger)>=required-accumulated)
                        dxx(i)=exp(logdiam)
                        if(i==3) return
                        i=i+1;required=requested(i)
                    enddo
                    accumulated=accumulated+fraction(trigger)
                else
                    stage(trigger)=1
                endif
            else
                stage(trigger)=stage(trigger)+1
            endif
        enddo
    end subroutine percentile_scan
end module braids_surface_identity_live_v69_module
