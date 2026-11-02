"""Generate a persistent-OpenMP-team version of the existing MIC-PCG filter.

Only auxiliary solver execution is changed; global/block preconditioners and
the full Helmholtz operator are retained. Generated Fortran is frozen by the
experiment builder. Both modes retain deterministic fixed-tile dot products.
"""
from .profile_current import replace_once


def persistent_source(text):
    text = replace_once(text, "case('block');requested_mode=2",
        "case('block');requested_mode=2\n    case('persistent');requested_mode=3\n    case('persistent_block');requested_mode=4")
    text = text.replace('parallel_mode==2', '(parallel_mode==2.or.parallel_mode==4)')
    text = replace_once(text, '  call precondition();p=z;rho=inner(r,z)',
        '  if(parallel_mode>=3) then\n   call persistent_solve()\n   return\n  endif\n'
        '  call precondition();p=z;rho=inner(r,z)')
    return replace_once(text, ' contains\n  function inner(x,y) result(value)',
                        ' contains\n' + SOLVER + '\n  function inner(x,y) result(value)')


SOLVER = '''  subroutine persistent_solve()
   integer :: iteration,c
   logical :: finished,failed
   finished=.false.;failed=.false.
   !$omp parallel default(shared) private(iteration,c)
   call team_precondition()
   !$omp do schedule(static)
   do c=1,nc
    p(c)=z(c)
   enddo
   !$omp end do
   call team_dot(r,z,rho)
   do iteration=1,500
    call team_multiply(p,ap)
    call team_dot(p,ap,denominator)
    !$omp single
    failed=denominator<=0.or..not.ieee_is_finite(denominator)
    if(.not.failed) alpha=rho/denominator
    !$omp end single
    if(failed) exit
    !$omp do schedule(static)
    do c=1,nc
     filtered(c)=filtered(c)+alpha*p(c);r(c)=r(c)-alpha*ap(c)
    enddo
    !$omp end do
    call team_dot(r,r,residual)
    !$omp single
    iterations=iteration;residual=sqrt(residual);finished=residual<=tolerance
    !$omp end single
    if(finished) exit
    call team_precondition()
    call team_dot(r,z,rhonew)
    !$omp single
    beta=rhonew/rho;rho=rhonew
    !$omp end single
    !$omp do schedule(static)
    do c=1,nc
     p(c)=z(c)+beta*p(c)
    enddo
    !$omp end do
   enddo
   if(finished.and..not.failed) then
    call team_multiply(filtered,ap)
    !$omp do schedule(static)
    do c=1,nc
     r(c)=rhs(c)-ap(c)
    enddo
    !$omp end do
    call team_dot(r,r,residual)
    !$omp single
    residual=sqrt(residual)
    ok=residual<=10*tolerance.and.all(ieee_is_finite(filtered))
    !$omp end single
   endif
   !$omp end parallel
  end subroutine

  subroutine team_dot(x,y,value)
   real(8),intent(in) :: x(nc),y(nc)
   real(8),intent(out) :: value
   integer :: b,lo,hi
   !$omp do schedule(static)
   do b=1,size(dots)
    lo=1+(b-1)*512;hi=min(nc,b*512)
    dots(b)=sum(x(lo:hi)*y(lo:hi))
   enddo
   !$omp end do
   !$omp single
   value=sum(dots)
   !$omp end single
  end subroutine

  subroutine team_multiply(x,y)
   real(8),intent(in) :: x(nc)
   real(8),intent(out) :: y(nc)
   real(8) :: flux
   integer :: c,g,f,n
   !$omp do schedule(static)
   do c=1,nc
    y(c)=area(c)*x(c)
    do g=gather_row(c),gather_row(c+1)-1
     f=gather_face(g);n=neighbour(f)
     if(conductance(f)==0) cycle
     if(owner(f)==c) then
      flux=conductance(f)*(x(c)-x(n));y(c)=y(c)+flux
     else
      flux=conductance(f)*(x(owner(f))-x(c));y(c)=y(c)-flux
     endif
    enddo
   enddo
   !$omp end do
  end subroutine

  subroutine team_precondition()
   integer :: c,k,b,first,last
   if(.not.use_ic) then
    !$omp do schedule(static)
    do c=1,nc
     z(c)=r(c)/diagonal(c)
    enddo
    !$omp end do
    return
   endif
   !$omp do schedule(static)
   do b=1,blocks
    first=1;last=nc
    if(parallel_mode==4) then
     first=partition_start(b);last=partition_start(b+1)-1
    endif
    z(first:last)=r(first:last)
    do c=first,last
     do k=row(c),row(c+1)-1
      if(column(k)<first) cycle
      z(c)=z(c)-factor(k)*z(column(k))
     enddo
     z(c)=z(c)/pivot(c)
    enddo
    do c=last,first,-1
     z(c)=z(c)/pivot(c)
     do k=row(c),row(c+1)-1
      if(column(k)<first) cycle
      z(column(k))=z(column(k))-factor(k)*z(c)
     enddo
    enddo
   enddo
   !$omp end do
  end subroutine
'''
