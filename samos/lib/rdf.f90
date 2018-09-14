SUBROUTINE calculate_rdf(&
        positions, istart, istop, stepsize,      & !, stepsize &
        radius, density, cell, invcell , indices1, indices2, nbins, rdf, integral, radii, &
        nstep, nat, nat1, nat2 &
    )
    IMPLICIT NONE

    integer, intent(in)         ::          istart, istop, stepsize ! The ionic step I start and I end
    real*8, intent(in)          ::          cell(3,3), invcell(3,3)
    
    integer, intent(in)         ::          nat, nat1, nat2  ! The total number of atoms, and number of atoms that I calculate the den
    integer, intent(in)         ::          nstep, nbins  ! The total number of atoms, and number of atoms that I calculate the den
    real*8, intent(in)          ::          positions(nstep, nat,3)

    integer, intent(in)         ::          indices1(nat1) ! The indices of the atoms that I calculate the density from 
    integer, intent(in)         ::          indices2(nat2) ! The indices of the atoms that I calculate the density from 

    real*8, intent(in)          ::          radius ! The maximum radius I should go for
    real*8, intent(in)          ::          density ! The density of the atoms
    real, parameter             ::          pi = 3.1415927

    real*8, intent(out) :: rdf(nbins), integral(nbins), radii(nbins) ! That's where I count
    real*8 :: distance_real(1:3), distance_crystal(1:3)
    real*8 :: binsize, distance_norm, vb, factor, r

    integer :: iat1, iat2, istep, bin, idim !, idim, iat
    ! This is the amount I augment the counter by, so that I correctly
    ! mean over the steps and the number of atoms
    factor = 1.0D0 * DBLE(stepsize) / DBLE(istop+1-istart)  / DBLE(nat1)
    rdf(:) = 0.0D0
    binsize = radius / DBLE (nbins)
    do istep=istart, istop, stepsize
        do iat1=1,nat1
            do iat2=1, nat2
                if ( indices2(iat2) .eq. indices1(iat1) ) CYCLE
                ! I calculate the distance between atom1 and atom2 in real space:
                distance_real(1:3) = positions(istep, indices2(iat2), 1:3) - &
                        positions(istep, indices1(iat1), 1:3)
                ! I multiply with invcell to calculate the distance
                ! in crystal coordinates, and take the modulo 1 to get the distance
                ! vector into the unit cell
                distance_crystal(1:3) = MOD(MATMUL(invcell, distance_real), 1.0)
                ! The following gets the shortest distance of atom1 to atom2 in
                ! an orthorhombic system. For accute cells, this might be wrong.
                DO idim=1, 3
                    IF ( distance_crystal(idim) < -0.5 ) THEN
                        distance_crystal(idim) = distance_crystal(idim) + 1.0D0
                    ELSEIF ( distance_crystal(idim) > 0.5 ) THEN
                        distance_crystal(idim) = 1.0D0 - distance_crystal(idim)
                    ENDIF
                ENDDO
                distance_real(1:3) = MATMUL(cell, distance_crystal)
                ! Calculate the norm of vector
                distance_norm = SQRT(SUM(distance_real(1:3)*distance_real(1:3)))
                ! and in which bin it belongs:
                ! Here I take the nearest bin.
                bin = nint( distance_norm / binsize )
                if ( bin > 0 .and. bin <= nbins ) rdf(bin) = rdf(bin)  + factor
            enddo
        end do
    enddo
    integral(1) = 0.0D0
    bin = 1
    integral(bin) = 0.0D0
    ! Below expression calculates the volume element for a certain bin,
    ! in this case the first:
    factor = 4.0D0/3.0D0 * pi
    vb = factor *( ((bin+0.5)*binsize)**3 - ((bin-0.5D0)*binsize)**3 )
    ! I normalize the RDF by the ideal gas density:
    rdf(bin) = rdf(bin) / (vb * density)
    radii(bin) = binsize
    do bin=2, nbins
        integral(bin) = integral(bin-1) + rdf(bin)
        r = DBLE(bin) * binsize
        radii(bin) = r
        vb = factor * (3.0D0 * r**2 *binsize + 0.75D0 * binsize**3)
        rdf(bin) = rdf(bin) / (vb * density) ! number of particles in a real gas
    end do

END SUBROUTINE calculate_rdf

