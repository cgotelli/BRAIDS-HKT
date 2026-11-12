module braids_viscous_inlet_module
 use braids_hydraulic_roughness_module, only: braids_roughness_state, &
  braids_roughness_manning
 use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
 implicit none
 private

 integer,parameter,public :: BRAIDS_VISCOUS_INLET_SUCCESS=0
 integer,parameter,public :: BRAIDS_VISCOUS_INLET_INVALID=1
 integer,parameter,public :: BRAIDS_VISCOUS_INLET_NO_WET_SECTION=2
 integer,parameter,public :: BRAIDS_VISCOUS_INLET_NO_DRAG=3
 integer,parameter,public :: BRAIDS_VISCOUS_INLET_NONCONVERGED=4
 public :: braids_viscous_manning_profile,braids_viscous_roughness_profile

contains

 subroutine braids_viscous_manning_profile(number,width,center,depth,roughness, &
   viscosity,discharge,gravity,dry_tolerance,work_depth,velocity,driving, &
   component,iterations,discharge_residual,maximum_acceleration_residual,status)
  ! Frozen compatibility wrapper for v39--v53 decks.  The generic operator
  ! below consumes the same raw typed roughness parameter as hydraulic
  ! friction; this wrapper preserves the historical Manning ABI exactly.
  integer,intent(in) :: number
  real(8),intent(in) :: width(number),center(number),depth(number),roughness(number)
  real(8),intent(in) :: viscosity,discharge,gravity,dry_tolerance,work_depth
  real(8),intent(out) :: velocity(number),driving,discharge_residual
  real(8),intent(out) :: maximum_acceleration_residual
  integer,intent(out) :: component(number),iterations,status

  call braids_viscous_roughness_profile(number,width,center,depth,roughness, &
   braids_roughness_manning,.false.,viscosity,discharge,gravity,dry_tolerance, &
   work_depth,velocity,driving,component,iterations,discharge_residual, &
   maximum_acceleration_residual,status)
 end subroutine

 subroutine braids_viscous_roughness_profile(number,width,center,depth,roughness, &
   roughness_mode,white_colebrook_chzk2d,viscosity,discharge,gravity,dry_tolerance, &
   work_depth,velocity,driving,component,iterations,discharge_residual, &
   maximum_acceleration_residual,status)
  ! Steady cross-section balance for the velocity-Laplacian and Manning
  ! or typed C/M/W source used by BRAIDS:
  !
  !   nu L_N u + B beta u|u| = B G,  sum(B h u)=Q,
  !   beta=(g/C(h,p)^2)/h.
  !
  ! Boundary fragments are sorted deterministically along the supplied
  ! tangent coordinate. Dry fragments and physical gaps split the free-slip
  ! Neumann graph, while every wet component shares one driving acceleration
  ! G. The bordered Newton system is solved in O(N) with two tridiagonal
  ! solves. No previous-step state is used, preserving retry/restart identity.
  integer,intent(in) :: number
  real(8),intent(in) :: width(number),center(number),depth(number),roughness(number)
  real(8),intent(in) :: viscosity,discharge,gravity,dry_tolerance,work_depth
  integer,intent(in) :: roughness_mode
  logical,intent(in) :: white_colebrook_chzk2d
  real(8),intent(out) :: velocity(number),driving,discharge_residual
  real(8),intent(out) :: maximum_acceleration_residual
  integer,intent(out) :: component(number),iterations,status
  integer,allocatable :: order(:),wet_position(:),wet_component(:)
  real(8),allocatable :: sorted_width(:),sorted_center(:),sorted_depth(:),sorted_roughness(:)
  real(8),allocatable :: wet_width(:),wet_depth(:),wet_roughness(:),beta(:),weight(:)
  real(8),allocatable :: edge(:),u(:),residual(:),acceleration(:),diagonal(:),lower(:),upper(:)
  real(8),allocatable :: primary(:),sensitivity(:),delta_u(:),candidate_u(:)
  logical,allocatable :: touching(:),wet(:)
  real(8) :: scale,geometry_tolerance,gap,distance,wet_area
  real(8) :: constraint,acceleration_scale,merit,qscale,q_tolerance,acceleration_tolerance
  real(8) :: denominator,delta_driving,alpha,candidate_driving,candidate_constraint
  real(8) :: candidate_scale,candidate_merit,pivot_tolerance,local_chezy,local_z0,local_drag
  integer :: i,j,k,m,ncomponent,local_iteration,line_search,solve_status
  logical :: converged,accepted,anchored,roughness_valid
  real(8),parameter :: relative_tolerance=2.d-11
  real(8),parameter :: absolute_acceleration_tolerance=2.d-14

  velocity=0.d0;component=-1;driving=0.d0;iterations=0
  discharge_residual=0.d0;maximum_acceleration_residual=0.d0
  status=BRAIDS_VISCOUS_INLET_INVALID
  if(number<1) return
  if(.not.all(ieee_is_finite(width)).or..not.all(ieee_is_finite(center)).or. &
     .not.all(ieee_is_finite(depth)).or..not.all(ieee_is_finite(roughness)).or. &
     .not.all(ieee_is_finite([viscosity,discharge,gravity,dry_tolerance,work_depth]))) return
  if(any(width<=0.d0).or.any(depth<0.d0).or.any(roughness<0.d0).or. &
     viscosity<0.d0.or.discharge<0.d0.or.gravity<=0.d0.or. &
     dry_tolerance<=0.d0.or.work_depth<0.d0) return

  allocate(order(number),sorted_width(number),sorted_center(number), &
   sorted_depth(number),sorted_roughness(number),wet(number),touching(max(0,number-1)))
  do i=1,number
   order(i)=i
  enddo
  ! Stable insertion sort makes the result independent of input face order.
  do i=2,number
   k=order(i);j=i-1
   do while(j>=1)
    if(center(order(j))<=center(k)) exit
    order(j+1)=order(j);j=j-1
   enddo
   order(j+1)=k
  enddo
  do i=1,number
   sorted_width(i)=width(order(i));sorted_center(i)=center(order(i))
   sorted_depth(i)=depth(order(i));sorted_roughness(i)=roughness(order(i))
  enddo
  ! Base the geometric tolerance on the original projected-coordinate scale.
  ! Removing a common origin does not remove the ulp-scale error already
  ! carried by O(1e6 m) face centres.
  scale=max(1.d0,maxval(abs(sorted_center)),maxval(sorted_width))
  geometry_tolerance=512.d0*epsilon(1.d0)*scale
  sorted_center=sorted_center-sorted_center(1)
  if(number>1) then
   do i=1,number-1
    distance=sorted_center(i+1)-sorted_center(i)
    gap=distance-.5d0*(sorted_width(i)+sorted_width(i+1))
    if(distance<=0.d0.or.gap < -geometry_tolerance) then
     deallocate(order,sorted_width,sorted_center,sorted_depth,sorted_roughness,wet,touching)
     return
    endif
    touching(i)=gap<=geometry_tolerance
   enddo
  endif
  wet=sorted_depth>dry_tolerance
  m=count(wet)
  if(discharge>0.d0.and.m==0) then
   status=BRAIDS_VISCOUS_INLET_NO_WET_SECTION
   deallocate(order,sorted_width,sorted_center,sorted_depth,sorted_roughness,wet,touching)
   return
  endif
  if(discharge<=0.d0.or.m==0) then
   status=BRAIDS_VISCOUS_INLET_SUCCESS
   deallocate(order,sorted_width,sorted_center,sorted_depth,sorted_roughness,wet,touching)
   return
  endif

  allocate(wet_position(m),wet_component(m),wet_width(m),wet_depth(m), &
   wet_roughness(m),beta(m),weight(m),edge(max(0,m-1)),u(m),residual(m), &
   acceleration(m),diagonal(m),lower(m),upper(m),primary(m),sensitivity(m), &
   delta_u(m),candidate_u(m))
  j=0;ncomponent=0
  do i=1,number
   if(.not.wet(i)) cycle
   j=j+1;wet_position(j)=i
   if(i==1.or..not.wet(max(1,i-1))) then
    ncomponent=ncomponent+1
   else if(.not.touching(i-1)) then
    ncomponent=ncomponent+1
   endif
   wet_component(j)=ncomponent
   component(order(i))=ncomponent-1
   wet_width(j)=sorted_width(i);wet_depth(j)=sorted_depth(i)
   wet_roughness(j)=sorted_roughness(i)
   if(roughness_mode==braids_roughness_manning) then
    ! Keep the historical evaluation order for old decks/restarts.
    beta(j)=gravity*wet_roughness(j)**2/ &
     (wet_depth(j)*max(wet_depth(j),work_depth)**(1.d0/3.d0))
    roughness_valid=ieee_is_finite(beta(j)).and.beta(j)>0.d0
   else
    call braids_roughness_state(roughness_mode,wet_depth(j),wet_roughness(j), &
     gravity,work_depth,white_colebrook_chzk2d,local_chezy,local_z0,local_drag, &
     roughness_valid)
    if(roughness_valid) beta(j)=local_drag/wet_depth(j)
   endif
   if(.not.roughness_valid) beta(j)=0.d0
   weight(j)=wet_width(j)*wet_depth(j)
  enddo
  if(.not.all(ieee_is_finite(beta))) then
   call cleanup();return
  endif
  do k=1,ncomponent
   anchored=.false.
   do j=1,m
    if(wet_component(j)==k.and.beta(j)>0.d0) anchored=.true.
   enddo
   if(.not.anchored) then
    status=BRAIDS_VISCOUS_INLET_NO_DRAG;call cleanup();return
   endif
  enddo
  if(viscosity<=0.d0.and.any(beta<=0.d0)) then
   status=BRAIDS_VISCOUS_INLET_NO_DRAG;call cleanup();return
  endif

  if(m>1) then
   edge=0.d0
   do j=1,m-1
    if(wet_position(j+1)/=wet_position(j)+1) cycle
    if(wet_component(j+1)/=wet_component(j)) cycle
    distance=sorted_center(wet_position(j+1))-sorted_center(wet_position(j))
    if(distance<=0.d0) then
     call cleanup();return
    endif
    edge(j)=viscosity/distance
   enddo
  endif
  wet_area=sum(weight)
  if(wet_area<=0.d0.or..not.ieee_is_finite(wet_area)) then
   status=BRAIDS_VISCOUS_INLET_NO_WET_SECTION;call cleanup();return
  endif
  u=discharge/wet_area
  driving=sum(wet_width*beta*u*u)/sum(wet_width)
  if(driving<=0.d0.or..not.ieee_is_finite(driving)) then
   status=BRAIDS_VISCOUS_INLET_NO_DRAG;call cleanup();return
  endif

  converged=.false.;pivot_tolerance=128.d0*tiny(1.d0)
  do local_iteration=0,80
   iterations=local_iteration
   call evaluate(u,driving,residual,constraint,acceleration,acceleration_scale,merit)
   qscale=max(discharge,abs(sum(weight*u)),tiny(1.d0))
   q_tolerance=128.d0*epsilon(1.d0)*qscale
   acceleration_tolerance=absolute_acceleration_tolerance+relative_tolerance*acceleration_scale
   if(maxval(abs(acceleration))<=acceleration_tolerance.and.abs(constraint)<=q_tolerance) then
    converged=.true.;exit
   endif
   if(local_iteration==80) exit
   lower=0.d0;upper=0.d0
   do j=1,m
    diagonal(j)=2.d0*wet_width(j)*beta(j)*u(j)
    if(j>1) then
     lower(j)=-edge(j-1);diagonal(j)=diagonal(j)+edge(j-1)
    endif
    if(j<m) then
     upper(j)=-edge(j);diagonal(j)=diagonal(j)+edge(j)
    endif
   enddo
   call tridiagonal_solve(lower,diagonal,upper,-residual,primary,pivot_tolerance,solve_status)
   if(solve_status/=0) then
    status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
   endif
   call tridiagonal_solve(lower,diagonal,upper,wet_width,sensitivity,pivot_tolerance,solve_status)
   if(solve_status/=0) then
    status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
   endif
   denominator=sum(weight*sensitivity)
   if(denominator<=0.d0.or..not.ieee_is_finite(denominator)) then
    status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
   endif
   delta_driving=(-constraint-sum(weight*primary))/denominator
   delta_u=primary+sensitivity*delta_driving
   if(.not.all(ieee_is_finite(delta_u)).or..not.ieee_is_finite(delta_driving)) then
    status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
   endif
   alpha=1.d0
   do j=1,m
    if(delta_u(j)<0.d0) alpha=min(alpha,.99d0*(-u(j)/delta_u(j)))
   enddo
   if(delta_driving<0.d0) alpha=min(alpha,.99d0*(-driving/delta_driving))
   accepted=.false.
   do line_search=1,60
    candidate_u=u+alpha*delta_u;candidate_driving=driving+alpha*delta_driving
    if(all(candidate_u>0.d0).and.candidate_driving>0.d0) then
     call evaluate(candidate_u,candidate_driving,residual,candidate_constraint, &
      acceleration,candidate_scale,candidate_merit)
     if(candidate_merit<merit) then
      u=candidate_u;driving=candidate_driving;accepted=.true.;exit
     endif
    endif
    alpha=.5d0*alpha
   enddo
   if(.not.accepted) then
    status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
   endif
  enddo
  if(.not.converged) then
   status=BRAIDS_VISCOUS_INLET_NONCONVERGED;call cleanup();return
  endif
  call evaluate(u,driving,residual,constraint,acceleration,acceleration_scale,merit)
  do j=1,m
   velocity(order(wet_position(j)))=u(j)
  enddo
  discharge_residual=sum(weight*u)-discharge
  maximum_acceleration_residual=maxval(abs(acceleration))
  status=BRAIDS_VISCOUS_INLET_SUCCESS
  call cleanup()

 contains

  subroutine evaluate(value,gradient,value_residual,value_constraint, &
    value_acceleration,value_scale,value_merit)
   real(8),intent(in) :: value(m),gradient
   real(8),intent(out) :: value_residual(m),value_constraint,value_acceleration(m)
   real(8),intent(out) :: value_scale,value_merit
   real(8) :: diffusion(m),local_qscale
   integer :: index
   diffusion=0.d0
   do index=1,m-1
    diffusion(index)=diffusion(index)+edge(index)*(value(index)-value(index+1))
    diffusion(index+1)=diffusion(index+1)+edge(index)*(value(index+1)-value(index))
   enddo
   value_residual=diffusion+wet_width*beta*value*value-wet_width*gradient
   value_constraint=sum(weight*value)-discharge
   value_acceleration=value_residual/wet_width
   value_scale=max(abs(gradient),maxval(beta*value*value), &
    maxval(abs(diffusion/wet_width)),tiny(1.d0))
   local_qscale=max(discharge,abs(sum(weight*value)),tiny(1.d0))
   value_merit=sqrt(sum((value_acceleration/value_scale)**2)/real(m,8)+ &
    (value_constraint/local_qscale)**2)
  end subroutine

  subroutine cleanup()
   if(allocated(order)) deallocate(order,sorted_width,sorted_center,sorted_depth,sorted_roughness,wet,touching)
   if(allocated(wet_position)) deallocate(wet_position,wet_component,wet_width,wet_depth, &
    wet_roughness,beta,weight,edge,u,residual,acceleration,diagonal,lower,upper, &
    primary,sensitivity,delta_u,candidate_u)
  end subroutine

 end subroutine braids_viscous_roughness_profile

 subroutine tridiagonal_solve(lower,diagonal,upper,rhs,solution,pivot_tolerance,status)
  real(8),intent(in) :: lower(:),diagonal(:),upper(:),rhs(:),pivot_tolerance
  real(8),intent(out) :: solution(size(rhs))
  integer,intent(out) :: status
  real(8) :: cprime(size(rhs)),dprime(size(rhs)),pivot,scale
  integer :: i,n
  n=size(rhs);solution=0.d0;status=1
  if(n<1.or.size(lower)/=n.or.size(diagonal)/=n.or.size(upper)/=n) return
  scale=max(1.d0,maxval(abs(diagonal)),maxval(abs(lower)),maxval(abs(upper)))
  pivot=diagonal(1)
  if(.not.ieee_is_finite(pivot).or.pivot<=pivot_tolerance*scale) return
  cprime(1)=upper(1)/pivot;dprime(1)=rhs(1)/pivot
  do i=2,n
   pivot=diagonal(i)-lower(i)*cprime(i-1)
   if(.not.ieee_is_finite(pivot).or.pivot<=pivot_tolerance*scale) return
   cprime(i)=upper(i)/pivot;dprime(i)=(rhs(i)-lower(i)*dprime(i-1))/pivot
  enddo
  solution(n)=dprime(n)
  do i=n-1,1,-1
   solution(i)=dprime(i)-cprime(i)*solution(i+1)
  enddo
  if(.not.all(ieee_is_finite(solution))) then
   solution=0.d0;return
  endif
  status=0
 end subroutine

end module
