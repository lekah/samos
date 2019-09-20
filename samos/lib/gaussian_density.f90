! f2py -c -m gaussian_density gaussian_density.f90



SUBROUTINE make_gaussian_density(&
        positions, gridf, n1,n2,n3, b1,b2,b3, istart, istop, stepsize,      & !, stepsize &
        sigma, cell, invcell , indices_i_care, pbar_frequency, nstep,nat,  nat_this_species  &
    )
    ! https://tavianator.com/exact-bounding-boxes-for-spheres-ellipsoids/
    ! http://blog.yiningkarlli.com/2013/02/bounding-boxes-for-ellipsoids.html
    ! http://www.iquilezles.org/www/articles/ellipses/ellipses.htm

    ! The input:
    IMPLICIT NONE

    CHARACTER*256, intent(in)   ::          gridf ! The filename where I write the Grid
    integer, intent(in)         ::          n1,n2,n3 ! The number of gridpoints in each direction
    integer, intent(in)         ::          b1,b2,b3 ! The number of gridpoints that I walk for each poing
    ! integer, intent(in)         ::          n_sigma  ! How many , initial_step,    &
    integer, intent(in)         ::          istart, istop, stepsize ! The ionic step I start and I end
    real(kind=8), intent(in)    ::          cell(3,3)
    real(kind=8), intent(in)    ::          invcell(3,3)

    integer, intent(in)         ::          nat  ! The total number of atoms, and number of atoms that I calculate the den
    integer, intent(in)         ::          nstep  ! The total number of atoms, and number of atoms that I calculate the den
    real(kind=8), intent(in)    ::          positions(nstep, nat,3)
    ! logical, intent(in)         ::          recenter ! Whether to recenter


    integer, intent(in)         ::          indices_i_care(nat_this_species) ! The indices of the atoms that I calculate the density from
    integer, intent(in)         ::          pbar_frequency ! Frequency bar update

    integer, intent(in)         ::          nat_this_species   ! and number of atoms that I calculate the density of
    real(kind=8), intent(in)    ::          sigma   ! Sigma value
    ! real(kind=8), intent(in)    ::          box_a, box_b, box_c   ! The cell dimensions in angstrom, and the conversion from positions read to angstrom
    ! IMPORTANT! I removed recentering, this should be handled by a different function!

    real(kind=8), allocatable :: counted(:,:,:) ! That's where I count

    integer      :: cont, iat, istep, iat_i_care, in1, in2, in3, idir
    integer      :: jn1, jn2, jn3
    ! integer, allocatable :: natoms(:)
    real(kind=8) :: somma, dV
    real(kind=8) :: pos_grid_point_crystal(1:3), pos_grid_point_real(1:3), pos_atom_real(1:3), pos_atom_crystal(1:3)
    real(kind=8) :: get_from_gaussian
    integer ::      is, js, ks



    allocate(counted(n1,n2,n3))

    ! I calculate the total volume as the deteminant of the cell matrix.
    ! dV is that devided by number of gridpoints
    dV = abs((cell(1,1) * cell(2,2) * cell(3,3) &
         + cell(2,1) * cell(3,2) * cell(1,3) &
         + cell(3,1) * cell(1,3) * cell(2,3) &
         - cell(3,1) * cell(2,2) * cell(1,3) &
         - cell(2,1) * cell(1,2) * cell(3,3) &
         - cell(1,1) * cell(3,2) * cell(2,3)) / DBLE(n1*n2*n3))
    !
    counted(:,:,:)=0.d0

    do istep=istart, istop, stepsize
        IF ( MOD(istep, pbar_frequency) == 0) WRITE(*,'(A1)', advance='no') '='
!~         read(20,*)
        ! Now I only do stuff for the indices that I care about:
        do iat=1,nat_this_species
            iat_i_care = indices_i_care(iat)
            pos_atom_real(1:3) = positions(istep, iat_i_care, 1:3)
            ! Now, here I put into the crystal coordinates
            ! Applying modulo 1 to bring them into one of the 8 possible unit cells!
            pos_atom_crystal(1:3) =  MOD(MATMUL(invcell, pos_atom_real(1:3)),1.0D0)
            ! Translate by one unit cell if coordinate is negative!
            DO idir=1,3
                IF ( pos_atom_crystal(idir) < 0.0D0) pos_atom_crystal(idir) = 1.0D0 + pos_atom_crystal(idir)
            END DO
            ! For later, I calculate now the position (real) of the atom inside
            ! the unit cell
            pos_atom_real(1:3) = MATMUL(cell, pos_atom_crystal)
            ! Now I need to find the closest grid point in the cell.
            ! Since I have boxes n1,n2,n3 I am at
            in1 = int(pos_atom_crystal(1) * n1) + 1
            in2 = int(pos_atom_crystal(2) * n2) + 1
            in3 = int(pos_atom_crystal(3) * n3) + 1

            ! For each direction, I walk up and down from my starting bound
            ! How much I walk up and down is defined by the bounding box:
            do is = in1-b1, in1+b1
                ! Calculating the first coordinate of gridpoints
                ! Important: This can of course lie outside the unit cell, but
                ! that's the desired behavior
                pos_grid_point_crystal(1) = DBLE(is) / DBLE(n1)
                ! I apply the modulo and optionally translate to get the correct index
                jn1 = mod(is,n1)
                if (jn1 <= 0) jn1 = jn1 + n1
                do js = in2-b2 , in2+b2
                    ! As above for second coordinate:
                    pos_grid_point_crystal(2) = DBLE(js) / DBLE(n2)
                    jn2 = mod(js,n2)
                    if (jn2 <= 0) jn2 = jn2 + n2

                    do ks = in3-b3 , in3 + b3
                        ! As above for third coordinate
                        jn3 = mod(ks,n3)
                        if (jn3 <= 0) jn3 = jn3 + n3
                        pos_grid_point_crystal(3) = DBLE(ks) / DBLE(n3)

                        ! NUMBER CRUNCHING:
                        ! I calculate the grid point in real coordinates:
                        pos_grid_point_real(1:3) = MATMUL(cell, pos_grid_point_crystal)
                        ! I add the value taken from the gaussian based on the difference:
                        counted(jn1,jn2,jn3) = counted(jn1,jn2,jn3) + get_from_gaussian(&
                                sigma, &
                                pos_grid_point_real(1:3) - pos_atom_real(1:3) &
                            )
                    enddo
                enddo
            enddo
        enddo
    enddo

    ! I integrate counted (the gaussians) over the whole space:
    somma=0.d0
    do in1=1,n1
        do in2=1,n2
            do in3=1,n3
                somma=somma + counted(in1,in2,in3) * dV
            end do
        end do
    end do


    ! My wanted behavior is that the TOTAL sums to the atoms:
    do in1=1,n1
        do in2=1,n2
            do in3=1,n3
                counted(in1,in2,in3) = counted(in1,in2,in3) * nat_this_species / somma
            end do
        end do
    end do
    !

    ! N
    open(unit=21,file=gridf,status='old', access='append')
    cont=1
    do in3=0,n3
        do in2=0,n2
            do in1=0,n1
                if (cont<=5) then
                    write(21,'(f20.10)', advance='no')  counted(MOD(in1, n1)+1, MOD(in2, n2)+1, MOD(in3, n3)+1)
                    cont=cont+1
                else
                    write(21,'(f20.10)') counted(MOD(in1, n1)+1, MOD(in2, n2)+1, MOD(in3, n3)+1)
                    cont=1
                end if
            end do
        end do
    end do
    if (cont > 1) write(21, *)
    write(21, *) "   END_DATAGRID_3D"
    write(21, *) "END_BLOCK_DATAGRID_3D"
!~     deallocate(positions)
    deallocate(counted)

    close(21)
END SUBROUTINE make_gaussian_density



real(kind=8) function get_from_gaussian(sigma, vector )
!
    implicit none

    real(kind=8) :: sigma, vector(3)
    real(kind=8) :: sq_modulus
    !
    sq_modulus = vector(1) * vector(1) + vector(2) * vector(2) + vector(3) * vector(3)
    !
    get_from_gaussian = exp(- sq_modulus / (2.d0 * sigma * sigma) )

!
end function
