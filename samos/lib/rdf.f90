SUBROUTINE calculate_rdf(&
        positions, istart, istop, stepsize,      & !, stepsize &
        radius, cell, invcell , indices1, indices2, nbins, rdf, nstep, nat, nat1, nat2 &
    )


    ! The input:
    IMPLICIT NONE

    integer, intent(in)         ::          istart, istop, stepsize ! The ionic step I start and I end
    real*8, intent(in)    ::          cell(3,3), invcell(3,3)
    
    integer, intent(in)         ::          nat, nat1, nat2  ! The total number of atoms, and number of atoms that I calculate the den
    integer, intent(in)         ::          nstep, nbins  ! The total number of atoms, and number of atoms that I calculate the den
    real*8, intent(in)    ::          positions(nstep, nat,3)
    ! logical, intent(in)         ::          recenter ! Whether to recenter 


    integer, intent(in)         ::          indices1(nat1) ! The indices of the atoms that I calculate the density from 
    integer, intent(in)         ::          indices2(nat2) ! The indices of the atoms that I calculate the density from 

    real*8, intent(in)    ::          radius   ! Sigma value
    ! real(kind=8), intent(in)    ::          box_a, box_b, box_c   ! The cell dimensions in angstrom, and the conversion from positions read to angstrom
    ! IMPORTANT! I removed recentering, this should be handled by a different function!
    
    real*8, intent(out) :: rdf(nbins) ! That's where I count
    real*8 :: distance_real(1:3), distance_crystal(1:3)
    real*8 :: binsize, distance_norm

    integer :: iat1, iat2, istep, bin !, idim, iat

    rdf(:) = 0.0D0
    binsize = radius / DBLE (nbins) 
!~     print*, binsize
    do istep=istart, istop, stepsize
!~         print*, istep
!~         IF ( MOD(istep, pbar_frequency) == 0) WRITE(*,'(A1)', advance='no') '='
        ! Now I only do stuff for the indices that I care about:
        do iat1=1,nat1
            do iat2=1, nat2
                if ( indices2(iat2) .eq. indices1(iat1) ) CYCLE 
                distance_real(1:3) = positions(istep, indices2(iat2), 1:3) - &
                        positions(istep, indices1(iat1), 1:3) 
!~                 print*, indices1(iat1), indices2(iat2)
!~                 print*, 'real1', distance_real
                distance_crystal(1:3) = MOD(MATMUL(invcell, distance_real), 1.0D0)
!~                 print*, 'crystal', distance_crystal
                distance_real(1:3) = MATMUL(cell, distance_crystal)
!~                 print*, 'real2', distance_real
                distance_norm = SQRT(SUM(distance_real(1:3)*distance_real(1:3))) ! norm of vector
!~                 print*, distance_norm, binsize, int(distance_norm / binsize)

                IF ( distance_norm >= radius ) CYCLE
                bin = int(distance_norm / binsize) +1
!~                 print*, int(distance_norm / binsize)
                rdf(bin) = rdf(bin)  +1.0D0
            enddo
        end do
    enddo
!~     print*, rdf(:)

END SUBROUTINE calculate_rdf

