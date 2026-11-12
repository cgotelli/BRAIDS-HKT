! SPDX-License-Identifier: GPL-3.0-or-later
! Current-only Delft3D Van Rijn (1993) suspended-load equilibrium closure.
!
! The equations and fixed 20-layer 2DH column follow the Delft3D morphology
! routines bedbc1993, calseddf1993, tram1, and soursin_2d.  This BRAIDS
! implementation is kept cell-local so AMR patches may continue to run in
! parallel.  Waves, hiding/exposure, mud fractions, and bed load are outside
! this first closure.
! Derived from GPL-3.0-or-later Delft3D morphology code, copyright
! Stichting Deltares, 2011-2026; see the exact source paths in the verifier.
module braids_van_rijn_1993_module
    implicit none
    private
    public :: braids_vr1993_grain_parameters,braids_vr1993_coefficients

    integer, parameter :: vertical_layers=20
    real(kind=8), parameter :: von_karman=0.41d0
    real(kind=8), parameter :: kernel_epsilon=1.d-6
    real(kind=8), parameter :: euler_number=exp(1.d0)
    real(kind=8), parameter :: sigma(vertical_layers) = [ &
        -0.0874d0,-0.2472d0,-0.3797d0,-0.4897d0,-0.5809d0, &
        -0.6565d0,-0.7193d0,-0.7713d0,-0.8145d0,-0.8503d0, &
        -0.8800d0,-0.9046d0,-0.9250d0,-0.9419d0,-0.9560d0, &
        -0.9676d0,-0.9773d0,-0.9854d0,-0.9920d0,-0.9975d0]
    real(kind=8), parameter :: thickness(vertical_layers) = [ &
        0.1747d0,0.1449d0,0.1202d0,0.0997d0,0.0827d0, &
        0.0686d0,0.0569d0,0.0472d0,0.0391d0,0.0325d0, &
        0.0269d0,0.0223d0,0.0185d0,0.0154d0,0.0127d0, &
        0.0106d0,0.0088d0,0.0073d0,0.0060d0,0.0050d0]
contains
    pure subroutine braids_vr1993_grain_parameters(d50,grain_density,water_density, &
                                                   gravity,viscosity,dstar,tau_critical)
        real(kind=8), intent(in) :: d50,grain_density,water_density,gravity,viscosity
        real(kind=8), intent(out) :: dstar,tau_critical
        real(kind=8) :: shields
        ! Delft3D initsedtra uses 0.3333 rather than a rounded 1/3 exponent.
        dstar=d50*(((grain_density-water_density)/water_density)*gravity/viscosity**2)**0.3333d0
        if (dstar<=4.d0) then
            shields=0.24d0/dstar
        else if (dstar<=10.d0) then
            shields=0.14d0/dstar**0.64d0
        else if (dstar<=20.d0) then
            shields=0.04d0/dstar**0.10d0
        else if (dstar<=150.d0) then
            shields=0.013d0*dstar**0.29d0
        else
            shields=0.055d0
        endif
        tau_critical=(grain_density-water_density)*gravity*d50*shields
    end subroutine braids_vr1993_grain_parameters

    pure subroutine braids_vr1993_coefficients(h,hu,hv,settling_velocity,grain_density, &
                                               water_density,viscosity,d50,d90,z0_current, &
                                               aks_factor,max_volume_concentration, &
                                               adaptation_factor,use_delft_ws_floor, &
                                               dstar,tau_critical,equilibrium_concentration, &
                                               exchange_rate,ustar,tau_ratio,caks,aks,tsd)
        real(kind=8), intent(in) :: h,hu,hv,settling_velocity,grain_density,water_density
        real(kind=8), intent(in) :: viscosity,d50,d90,z0_current,aks_factor
        real(kind=8), intent(in) :: max_volume_concentration,adaptation_factor
        logical, intent(in) :: use_delft_ws_floor
        real(kind=8), intent(in) :: dstar,tau_critical
        real(kind=8), intent(out) :: equilibrium_concentration,exchange_rate
        real(kind=8), intent(out) :: ustar,tau_ratio,caks,aks,tsd
        integer :: k,kmaxsd
        real(kind=8) :: umod,zumod,rc,f1c,fc,muc,tauc,transport_stage
        real(kind=8) :: beta,epsmxc,zi,dz,diffbt,factor,bakdif
        real(kind=8) :: seddif(0:vertical_layers),profile(vertical_layers)
        real(kind=8) :: avgu,avgcu,z,ulog,wsl,u,w,x,x2,x3,hots

        equilibrium_concentration=0.d0; exchange_rate=0.d0
        ustar=0.d0; tau_ratio=0.d0; caks=0.d0; aks=0.d0; tsd=0.d0
        if (h<=0.d0) return
        umod=sqrt(hu*hu+hv*hv)/h
        zumod=h/euler_number
        rc=30.d0*z0_current
        aks=min(max(aks_factor*rc,0.01d0*h),0.2d0*h)
        ustar=umod*von_karman/log(1.d0+zumod/z0_current)
        tauc=water_density*ustar*ustar
        f1c=0.24d0*log10(12.d0*h/(3.d0*d90))**(-2)
        fc=0.24d0*log10(12.d0*h/rc)**(-2)
        muc=f1c/fc
        tau_ratio=muc*tauc/tau_critical
        transport_stage=max(0.d0,tau_ratio-1.d0)
        if (transport_stage>kernel_epsilon) &
            caks=min(0.015d0*d50*transport_stage**1.5d0/ &
                     (aks*dstar**0.3d0),max_volume_concentration)

        kmaxsd=1
        do k=vertical_layers-1,1,-1
            zi=(1.d0+sigma(k)-thickness(k)/2.d0)*h
            if (zi>=aks) then
                kmaxsd=k
                exit
            endif
        enddo

        if (ustar>kernel_epsilon) then
            beta=min(max(1.d0+2.d0*(settling_velocity/ustar)**2,1.d0),1.5d0)
        else
            beta=1.d0
        endif
        epsmxc=0.25d0*von_karman*beta*ustar*h
        seddif(0)=epsmxc
        do k=1,vertical_layers
            zi=(1.d0+sigma(k)-thickness(k)/2.d0)*h
            if (zi>=0.5d0*h) then
                seddif(k)=epsmxc
            else
                seddif(k)=epsmxc*(1.d0-(1.d0-2.d0*zi/h)**2)
            endif
        enddo

        profile=0.d0
        bakdif=viscosity
        if (caks>kernel_epsilon) then
            dz=h*(1.d0+sigma(kmaxsd))-aks
            diffbt=max(seddif(kmaxsd)+bakdif,0.1d0*settling_velocity*dz)
            factor=1.d0+dz*settling_velocity/diffbt
            profile(kmaxsd)=caks*grain_density/factor
            do k=kmaxsd-1,1,-1
                dz=h*(sigma(k)-sigma(k+1))
                diffbt=max(seddif(k)+bakdif,0.1d0*settling_velocity*dz)
                factor=1.d0+dz*settling_velocity/diffbt
                profile(k)=profile(k+1)/factor
            enddo
            do k=kmaxsd+1,vertical_layers
                profile(k)=profile(k-1)
            enddo
        endif

        avgu=0.d0; avgcu=0.d0
        if (zumod>0.d0) then
            do k=1,vertical_layers
                z=(1.d0+sigma(k))*h
                ulog=log(1.d0+z/z0_current)
                avgu=avgu+ulog*thickness(k)
                avgcu=avgcu+ulog*profile(k)*thickness(k)
            enddo
            equilibrium_concentration=avgcu/max(avgu,kernel_epsilon)
        endif

        if (use_delft_ws_floor) then
            wsl=max(1.d-3,settling_velocity)
        else
            wsl=settling_velocity
        endif
        if (wsl<=0.d0) return
        if (umod>kernel_epsilon.and.ustar>kernel_epsilon) then
            u=min(ustar/umod,0.15d0)
            w=min(wsl/ustar,1.d0)
            x=w; x2=x*x; x3=x2*x
            ! These coefficients are deliberately default-real first, as in
            ! Delft3D soursin_2d, then promoted to the working precision.
            tsd=x*exp((real(1.547,8)-real(20.12,8)*u)*x3+ &
                      (real(326.832,8)*u**real(2.2047,8)-real(0.2,8))*x2+ &
                      (real(0.1385,8)*log(u)-real(6.4061,8))*x+ &
                      (real(0.5467,8)*u+real(2.1963,8)))
            hots=wsl/(tsd*adaptation_factor)
            exchange_rate=hots/h
        else
            equilibrium_concentration=0.d0
            exchange_rate=wsl/h
        endif
    end subroutine braids_vr1993_coefficients
end module braids_van_rijn_1993_module
