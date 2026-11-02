! Pure parent-settling and endpoint exchange kernels; no accepted-state owner.
! Equations follow the BRAIDS locked FLOW2DH oracles, not FLOW discretisation.
module braids_settling_exchange_v97_module
  use iso_c_binding, only: c_int, c_double, c_float
  use ieee_arithmetic, only: ieee_is_finite
  use omp_lib, only: omp_in_parallel
  implicit none
  integer, parameter :: ledger_block_size = 1024
contains
  subroutine braids_parent_settling_v97(nc, ns, ng, concentration, diameter, solid_density, &
      water_density, temperature, wet, cref, gravity, threads, clear, velocity, total, &
      hindered, viscosity, status) bind(C)
    integer(c_int), value :: nc, ns, ng, threads
    real(c_double), intent(in) :: concentration(nc,ns), diameter(nc,ng), solid_density(ng)
    real(c_double), intent(in) :: water_density(nc), temperature(nc)
    integer(c_int), intent(in) :: wet(nc)
    real(c_double), value :: cref, gravity
    real(c_double), intent(out) :: clear(nc,ng), velocity(nc,ng), total(nc), hindered(nc), viscosity(nc)
    integer(c_int), intent(out) :: status
    integer :: cell, grain, block, bad
    real(c_double) :: d, relative, nu, coefficient, sum_c, ws, factor
    bad = 0
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(cell,grain,block,d,relative,nu,coefficient,sum_c,ws,factor) reduction(max:bad)
    do block=1,(nc+ledger_block_size-1)/ledger_block_size
      do cell=(block-1)*ledger_block_size+1,min(nc,block*ledger_block_size)
      sum_c = 0.0_c_double
      do grain=1,ns
        sum_c = sum_c + concentration(cell,grain)
      end do
      nu = 4.0e-5_c_double/(20.0_c_double+temperature(cell))
      factor = max(0.0_c_double,1.0_c_double-max(0.0_c_double,sum_c)/cref)**5
      total(cell)=sum_c; hindered(cell)=factor; viscosity(cell)=nu
      do grain=1,ng
        d=diameter(cell,grain); relative=solid_density(grain)/water_density(cell)
        if (d < 1.5_c_double*64.0e-6_c_double) then
          ws=(relative-1.0_c_double)*gravity*d**2/(18.0_c_double*nu)
        else if (d < 0.5_c_double*2.0e-3_c_double) then
          coefficient=10.0_c_double
          if (d < 2.0_c_double*64.0e-6_c_double) &
            coefficient=(-2.9912_c_double/64.0e-6_c_double)*d+15.9824_c_double
          ws=coefficient*nu/d*(sqrt(1.0_c_double+(relative-1.0_c_double)*gravity*d**3/ &
              (100.0_c_double*nu**2))-1.0_c_double)
        else
          ws=1.1_c_double*sqrt((relative-1.0_c_double)*gravity*d)
        end if
        clear(cell,grain)=ws
        velocity(cell,grain)=0.0_c_double
        if (wet(cell)/=0) velocity(cell,grain)=ws*factor
        if (.not.ieee_is_finite(ws) .or. .not.ieee_is_finite(velocity(cell,grain))) bad=1
      end do
      end do
    end do
    !$omp end parallel do
    status=bad
  end subroutine

  subroutine braids_exchange_dry_transfer_v97(nc, ng, suspended, bed, capacity, fraction, &
      speed, shear, h0, h1, settling, supplied_timescale, area, wet, dt, factsd, sedthr, &
      reference_floor, threads, next_suspended, next_bed, equilibrium, terms, transfer, &
      cleanup, final_suspended, final_bed, sediment_active, ledger, status) bind(C)
    integer(c_int), value :: nc, ng, threads
    real(c_double), intent(in) :: suspended(nc,ng), bed(nc,ng), capacity(nc,ng), fraction(nc,ng)
    real(c_double), intent(in) :: speed(nc), shear(nc), h0(nc), h1(nc), settling(nc,ng)
    real(c_double), intent(in) :: supplied_timescale(nc,ng), area(nc)
    integer(c_int), intent(in) :: wet(nc)
    real(c_double), value :: dt, factsd, sedthr, reference_floor
    real(c_double), intent(out) :: next_suspended(nc,ng), next_bed(nc,ng), equilibrium(nc,ng)
    ! terms(:,:,:) is Python (6,ng,nc): explicit, implicit, settling, total, velocity, tsd.
    real(c_double), intent(out) :: terms(nc,ng,6), transfer(nc,ng), cleanup(nc,ng)
    real(c_double), intent(out) :: final_suspended(nc,ng), final_bed(nc,ng), ledger(6,ng)
    integer(c_int), intent(out) :: sediment_active(nc), status
    real(c_double), allocatable :: correction(:,:), partial(:,:,:)
    integer :: cell, grain, block, blocks, first, last, bad
    real(c_double) :: safe0, safe1, eq, wsl, u, x, x2, x3, exponent, tsd, hots
    real(c_double) :: explicit, implicit, sink, total_sink, candidate, bounded, inventory
    logical :: active, flowing
    allocate(correction(nc,ng))
    bad=0
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(cell,grain,block,safe0,safe1,eq,wsl,u,x,x2,x3,exponent,tsd,hots,explicit,implicit, &
    !$omp sink,total_sink,candidate,bounded,inventory,active,flowing) reduction(max:bad)
    do block=1,(nc+ledger_block_size-1)/ledger_block_size
      do cell=(block-1)*ledger_block_size+1,min(nc,block*ledger_block_size)
      active=wet(cell)/=0 .and. h1(cell)>sedthr
      sediment_active(cell)=0
      safe0=1.0_c_double; safe1=1.0_c_double
      if (active) then
        sediment_active(cell)=1
        safe0=max(reference_floor,h0(cell)); safe1=max(reference_floor,h1(cell))
      end if
      flowing=speed(cell)>1.0e-6_c_double .and. shear(cell)>1.0e-6_c_double
      do grain=1,ng
        eq=capacity(cell,grain)/(speed(cell)+1.0e-6_c_double)/safe1*fraction(cell,grain)
        equilibrium(cell,grain)=eq
        wsl=max(1.0e-3_c_double,settling(cell,grain))
        tsd=supplied_timescale(cell,grain)
        if (tsd==0.0_c_double .and. flowing) then
          u=min(shear(cell)/speed(cell),0.15_c_double)
          x=1.0_c_double
          if (shear(cell)>wsl) x=wsl/shear(cell)
          x2=x*x; x3=x2*x
          exponent=(real(1.547_c_float,c_double)-real(20.12_c_float,c_double)*u)*x3 &
            +(real(326.832_c_float,c_double)*u**real(2.2047_c_float,c_double)-real(0.2_c_float,c_double))*x2 &
            +(real(0.1385_c_float,c_double)*log(u)-real(6.4061_c_float,c_double))*x &
            +(real(0.5467_c_float,c_double)*u+real(2.1963_c_float,c_double))
          tsd=x*exp(exponent)
        end if
        hots=0.0_c_double; explicit=0.0_c_double; implicit=0.0_c_double
        if (flowing) then
          if (tsd<=0.0_c_double .or. .not.ieee_is_finite(tsd)) then
            bad=1
          else
            hots=wsl/(tsd*factsd)
            explicit=eq*hots/safe0
            implicit=(hots-wsl)/safe1
          end if
        end if
        sink=wsl/safe1
        total_sink=sink+implicit
        if (total_sink < -1024.0_c_double*epsilon(total_sink)) bad=1
        total_sink=max(total_sink,0.0_c_double)
        if (.not.active) then
          explicit=0.0_c_double; implicit=0.0_c_double; sink=0.0_c_double
          total_sink=0.0_c_double; hots=0.0_c_double; tsd=0.0_c_double
        end if
        terms(cell,grain,1)=explicit; terms(cell,grain,2)=implicit
        terms(cell,grain,3)=sink; terms(cell,grain,4)=total_sink
        terms(cell,grain,5)=hots; terms(cell,grain,6)=tsd
        inventory=suspended(cell,grain)+bed(cell,grain)
        candidate=(suspended(cell,grain)+dt*h0(cell)*explicit)/(1.0_c_double+dt*total_sink)
        if (.not.active) candidate=suspended(cell,grain)
        bounded=min(max(candidate,0.0_c_double),inventory)
        next_suspended(cell,grain)=bounded
        next_bed(cell,grain)=inventory-bounded
        transfer(cell,grain)=bounded-suspended(cell,grain)
        correction(cell,grain)=abs(bounded-candidate)
        cleanup(cell,grain)=0.0_c_double
        if (wet(cell)==0) cleanup(cell,grain)=bounded
        final_suspended(cell,grain)=bounded-cleanup(cell,grain)
        final_bed(cell,grain)=next_bed(cell,grain)+cleanup(cell,grain)
        if (.not.ieee_is_finite(candidate) .or. .not.ieee_is_finite(eq) .or. &
            .not.ieee_is_finite(total_sink) .or. .not.ieee_is_finite(tsd)) bad=1
      end do
      end do
    end do
    !$omp end parallel do
    ! Fixed blocks avoid atomics, thread-order dependence and serial cell reductions.
    blocks=(nc+ledger_block_size-1)/ledger_block_size
    allocate(partial(6,blocks,ng))
    !$omp parallel do collapse(2) schedule(static) if(.not.omp_in_parallel()) num_threads(threads) &
    !$omp private(grain,block,first,last,cell)
    do grain=1,ng
      do block=1,blocks
        partial(:,block,grain)=0.0_c_double
        first=(block-1)*ledger_block_size+1; last=min(nc,block*ledger_block_size)
        do cell=first,last
          partial(1,block,grain)=partial(1,block,grain)+transfer(cell,grain)*area(cell)
          partial(2,block,grain)=partial(2,block,grain)+(suspended(cell,grain)+bed(cell,grain))*area(cell)
          partial(3,block,grain)=partial(3,block,grain)+(next_suspended(cell,grain)+next_bed(cell,grain))*area(cell)
          partial(4,block,grain)=partial(4,block,grain)+correction(cell,grain)*area(cell)
          partial(5,block,grain)=partial(5,block,grain)+cleanup(cell,grain)*area(cell)
          partial(6,block,grain)=partial(6,block,grain)+(final_suspended(cell,grain)+final_bed(cell,grain))*area(cell)
        end do
      end do
    end do
    !$omp end parallel do
    !$omp parallel do schedule(static) if(.not.omp_in_parallel()) num_threads(threads) private(grain,block)
    do grain=1,ng
      ledger(:,grain)=0.0_c_double
      do block=1,blocks
        ledger(:,grain)=ledger(:,grain)+partial(:,block,grain)
      end do
    end do
    !$omp end parallel do
    deallocate(correction,partial)
    status=bad
  end subroutine
end module
