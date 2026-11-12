! Instantaneous equilibrium bedform roughness following the locked Delft3D4
! Fredsoe-MPM height, Van Rijn 1984 length, and Van Rijn 1984 roughness path.
module braids_bedform_roughness_module
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    implicit none
    private
    public :: braids_fredsoe_vanrijn84_roughness
contains
    pure subroutine braids_fredsoe_vanrijn84_roughness(h,u,v,seed_manning,d50,d90, &
            relative_density,critical_shields,effective_manning,dune_height,dune_length, &
            grain_ks,dune_ks,total_ks,iterations,valid)
        real(8),intent(in) :: h,u,v,seed_manning,d50,d90,relative_density,critical_shields
        real(8),intent(out) :: effective_manning,dune_height,dune_length,grain_ks,dune_ks,total_ks
        integer,intent(out) :: iterations
        logical,intent(out) :: valid
        real(8) :: speed,chezy,grain_chezy,mu,theta,next_manning,scale
        integer :: iteration
        effective_manning=seed_manning;dune_height=0;dune_length=0
        grain_ks=0;dune_ks=0;total_ks=0;iterations=0
        valid=all(ieee_is_finite([h,u,v,seed_manning,d50,d90,relative_density,critical_shields])).and. &
              h>0.d0.and.seed_manning>0.d0.and.d50>0.d0.and.d90>0.d0.and. &
              relative_density>0.d0.and.critical_shields>=0.d0
        if(.not.valid) return
        speed=sqrt(u*u+v*v);grain_ks=3.d0*d90;dune_length=7.3d0*h
        ! Delft3D feeds the roughness-updated Chezy back into the equilibrium
        ! height predictor on subsequent flow steps. Solve that local equilibrium
        ! directly. Starting from configured Manning selects the physically
        ! continuous branch without adding hidden state, preserving AMR/restart.
        do iteration=1,100
            chezy=h**(1.d0/6.d0)/effective_manning
            grain_chezy=18.d0*log10(12.d0*h/grain_ks)
            if(min(chezy,grain_chezy)<=0.d0) then
                valid=.false.;return
            endif
            mu=(chezy/grain_chezy)**1.5d0
            theta=speed*speed/(chezy*chezy*relative_density*d50)
            if(theta*mu>tiny(1.d0)) &
                dune_height=max((24.d0/63.d0)*(1.d0-critical_shields/(theta*mu))*h,0.d0)
            dune_ks=1.1d0*dune_height*(1.d0-exp(-25.d0*dune_height/dune_length))
            total_ks=min(.5d0*h,sqrt(grain_ks*grain_ks+dune_ks*dune_ks))
            total_ks=max(total_ks,tiny(1.d0))
            chezy=18.d0*log10(12.d0*h/total_ks)
            if(chezy<=0.d0) then
                valid=.false.;return
            endif
            next_manning=h**(1.d0/6.d0)/chezy
            scale=max(1.d0,abs(next_manning),abs(effective_manning))
            if(abs(next_manning-effective_manning)<=32.d0*epsilon(1.d0)*scale) then
                effective_manning=next_manning;iterations=iteration;exit
            endif
            effective_manning=.5d0*(effective_manning+next_manning)
        enddo
        if(iterations==0) then
            valid=.false.;return
        endif
        valid=all(ieee_is_finite([effective_manning,dune_height,dune_length,grain_ks,dune_ks,total_ks])).and. &
              effective_manning>0.d0.and.min(dune_height,dune_length,grain_ks,dune_ks,total_ks)>=0.d0
    end subroutine
end module
