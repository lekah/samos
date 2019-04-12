 ! fortran   f90
 
 ! To use it with python, compile with f2py



 
SUBROUTINE recenter_positions(positions, masses, factors, positions_, nstep, nat)
    implicit none
    INTEGER, intent(in)                             ::  nstep, nat
    ! masses are the masses of the atoms, factors are an additional factor so that we can
    ! calculate the center of mass only for certain species
    REAL*8, intent(in), dimension(nat)              ::  masses
    INTEGER, intent(in), dimension(nat)             :: factors
    REAL*8, intent(in),  dimension(nstep, nat, 3)   :: positions
    REAL*8, intent(out), dimension(nstep, nat, 3)   :: positions_

    INTEGER                                         :: iat, istep, ipol
    REAL*8, dimension(nat)                          :: rel_masses
    REAL*8, dimension(3)                            :: com
    REAL*8                                          ::  M

    DO iat=1, nat
        rel_masses(iat) = DBLE(factors(iat)) * masses(iat)
    END DO

    M = sum(rel_masses)
    rel_masses(:) = rel_masses(:) / M
    DO istep=1, nstep
        com(1:3) = 0.D0
        DO iat=1, nat
            DO ipol=1, 3
                com(ipol) = com(ipol) + rel_masses(iat)*positions(istep, iat, ipol) ! / M
            END DO
        END DO

        DO iat=1, nat
            DO ipol=1, 3
                positions_(istep, iat, ipol) = positions(istep, iat, ipol)  - com(ipol)
            END DO
        END DO
    END DO
END SUBROUTINE recenter_positions
 

SUBROUTINE recenter_velocities(velocities, masses, factors, velocities_, nstep, nat)
    implicit none
    ! Make the center of mass velocity disappear
    INTEGER, intent(in)                             ::  nstep, nat
    REAL*8, intent(in), dimension(nat)              :: masses
    INTEGER, intent(in), dimension(nat)             :: factors
    REAL*8, intent(in),  dimension(nstep, nat, 3)   :: velocities
    REAL*8, intent(out), dimension(nstep, nat, 3)   :: velocities_
    INTEGER                                         :: iat, istep, ipol
    REAL*8, dimension(nat)                          :: rel_masses
    REAL*8, dimension(3)                            :: com
    REAL*8                                          :: M

    DO iat=1, nat
        rel_masses(iat) = DBLE(factors(iat)) * masses(iat)
    END DO

    M = sum(rel_masses)
    rel_masses(:) = rel_masses(:) / M

    DO istep=1, nstep
        com(1:3) = 0.D0
        DO iat=1, nat
            DO ipol=1, 3
                com(ipol) = com(ipol) + rel_masses(iat)*velocities(istep, iat, ipol)
            END DO
        END DO

        DO iat=1, nat
            DO ipol=1, 3
                velocities_(istep, iat, ipol) = velocities(istep, iat, ipol)  - com(ipol)
            END DO
        END DO
    END DO

END SUBROUTINE recenter_velocities

SUBROUTINE calculate_msd_specific_atoms(    &
        positions,                              &
        indices_of_interest,                    &
        msd,                                    &
        stepsize_t,                             &
        stepsize_tau,                           &
        block_length_dt,                        &
        nr_of_blocks,                           &
        nr_of_t,                                &
        nstep,                                  &
        nat,                                    &
        nat_of_interest                         &
    )

    ! Fastest routine, instead of manipulating the trajectories 
    ! and passing only the trajectories of interest, 
    ! pass the indices of interest (remember index counting starts at 1)
    ! and let fortran do the magic
    ! Faster (x2) because no array manipulation is necessary

    implicit none

    INTEGER, intent(in)         ::  nstep,  nat,            &   ! Number of timesteps, number of atoms
                                    nat_of_interest,        &   ! number of atoms of interest
                                    nr_of_blocks,           &   ! Nr of blocks to calculate
                                    block_length_dt,        &   ! block length in dt
                                    nr_of_t,                &   ! nr of times of t
                                    stepsize_t,             &   ! The stepsize, eg 10 to calculate every 10th time
                                    stepsize_tau              ! The inner stepsize

    REAL*8, intent(in),  dimension(nstep, nat, 3)           :: positions
    INTEGER, intent(in), dimension(nat_of_interest)         :: indices_of_interest
    REAL*8, intent(out), dimension(nr_of_blocks, nr_of_t)   :: msd

    ! Local variables:
    INTEGER                     ::  t, tau,                    &        ! Time variables
                                    iat, ipol,                  &        ! Atom count
                                    iat_of_interest,           &        ! Atom count in indices_of interest
                                    iblock                              ! block count
    REAL*8                      ::  msd_this_t !, M

    ! Loop over the block
    DO iblock = 1, nr_of_blocks
        ! Loop over time t
        DO t = 1, nr_of_t
            ! Calculate one value of msd for this t in this block 
            ! based on the average over atoms and running window defined by tau
            msd_this_t = 0.0d0
            ! loop over atoms (remember this code only deals with single species
            DO iat_of_interest=1, nat_of_interest
                iat = indices_of_interest(iat_of_interest)
                ! Running average achieved by letting tau run through the whole block
                DO tau = (iblock -1)*block_length_dt+1, iblock*block_length_dt, stepsize_tau
                    DO ipol=1, 3
                        msd_this_t = msd_this_t + (positions(tau+stepsize_t*t, iat, ipol)-positions(tau, iat, ipol))**2
                    END DO
                END DO
            END DO
            msd(iblock, t) = msd_this_t
        END DO
    END DO
    msd(:,:) = msd(:, :) / DBLE(block_length_dt) / DBLE(nat_of_interest) * DBLE(stepsize_tau)  ! / DBLE(
end SUBROUTINE calculate_msd_specific_atoms


SUBROUTINE calculate_msd_specific_atoms_max_stats(    &
        positions,                              &
        indices_of_interest,                    &
        msd,                                    &
        stepsize_t,                             &
        stepsize_tau,                           &
        nr_of_t,                                &
        nstep,                                  &
        nat,                                    &
        nat_of_interest                         &
    )

    ! This subroutine does the same as calculate_msd_specific_atoms,
    ! but tries to maximize the statistics. It calculates a mean only,
    ! w/o blocking, but takes every datapoint for it's calculation!

    implicit none

    INTEGER, intent(in)         ::  nstep,  nat,            &   ! Number of timesteps, number of atoms
                                    nat_of_interest,        &   ! number of atoms of interest
                                    nr_of_t,                &   ! nr of times of t
                                    stepsize_t,             &   ! The stepsize, eg 10 to calculate every 10th time
                                    stepsize_tau                ! The inner stepsize

    REAL*8, intent(in),  dimension(nstep, nat, 3)           :: positions
    INTEGER, intent(in), dimension(nat_of_interest)         :: indices_of_interest
    REAL*8, intent(out), dimension(nr_of_t)   :: msd

    ! Local variables:
    INTEGER                     ::  t, tau,                    &        ! Time variables
                                    iat, ipol,                  &        ! Atom count
                                    iat_of_interest, &                      ! Atom count in indices_of interest
                                    icount

    REAL*8                      ::  msd_this_t, msd_this_t_tau_iat, fcount !, M

    ! Loop over time t. This here is the index that I will give,
    ! actual time is stepsize_t*t
    DO t = 1, nr_of_t
        ! Calculate one value of msd for this t in this block 
        ! based on the average over atoms and running window defined by tau
        msd_this_t = 0.0D0
        icount = 1
        DO iat_of_interest=1, nat_of_interest
            iat = indices_of_interest(iat_of_interest)
            ! Running average achieved by letting tau run through the entire trajectory!
            DO tau = 1, nstep - t*stepsize_t
                msd_this_t_tau_iat = 0.0D0
                ! You do not mean over directions!
                DO ipol=1, 3
                    msd_this_t_tau_iat = msd_this_t_tau_iat + &
                      (positions(tau+stepsize_t*t, iat, ipol)-positions(tau, iat, ipol))**2
                END DO
                fcount = DBLE(icount)
                msd_this_t = (fcount - 1.0D0 ) / fcount * msd_this_t + &
                        msd_this_t_tau_iat / fcount
                icount = icount + 1

            END DO
        END DO
        msd(t) = msd_this_t
        print*, t, msd_this_t
    END DO
    ! msd(:,:) = msd(:, :) / DBLE(nat_of_interest) * DBLE(stepsize_tau)  ! / DBLE(
end SUBROUTINE calculate_msd_specific_atoms_max_stats

SUBROUTINE get_com_positions(positions, masses, factors, positions_, nstep, nat)

    implicit none
    INTEGER, intent(in)                             ::  nstep, nat
    ! masses are the masses of the atoms, factors are an additional factor so that we can
    ! calculate the center of mass only for certain species
    REAL*8, intent(in), dimension(nat)              ::  masses
    INTEGER, intent(in), dimension(nat)             :: factors
    REAL*8, intent(in),  dimension(nstep, nat, 3)   :: positions
    REAL*8, intent(out), dimension(nstep, 1, 3)     :: positions_

    INTEGER                                         :: iat, istep, ipol
    REAL*8, dimension(nat)                          :: rel_masses
    REAL*8, dimension(3)                            :: com
    REAL*8                                          ::  M

    DO iat=1, nat
        rel_masses(iat) = DBLE(factors(iat)) * masses(iat)
    END DO

    M = sum(rel_masses)
    rel_masses(:) = rel_masses(:) / M
    DO istep=1, nstep
        com(1:3) = 0.D0
        DO iat=1, nat
            DO ipol=1, 3
                com(ipol) = com(ipol) + rel_masses(iat)*positions(istep, iat, ipol) ! / M
            END DO
        END DO
        positions_(istep, 1, 1:3) = com(1:3)
    END DO
END SUBROUTINE get_COM_positions


SUBROUTINE get_com_velocities(velocities, masses, factors, velocities_, nstep, nat)

    implicit none
    INTEGER, intent(in)                             ::  nstep, nat
    ! masses are the masses of the atoms, factors are an additional factor so that we can
    ! calculate the center of mass only for certain species
    REAL*8, intent(in), dimension(nat)              ::  masses
    INTEGER, intent(in), dimension(nat)             :: factors
    REAL*8, intent(in),  dimension(nstep, nat, 3)   :: velocities
    REAL*8, intent(out), dimension(nstep, 1, 3)     :: velocities_

    INTEGER                                         :: iat, istep, ipol
    REAL*8, dimension(nat)                          :: rel_masses
    REAL*8, dimension(3)                            :: com
    REAL*8                                          ::  M

    DO iat=1, nat
        rel_masses(iat) = DBLE(factors(iat)) * masses(iat)
    END DO

    M = sum(rel_masses)
    rel_masses(:) = rel_masses(:) / M
    DO istep=1, nstep
        com(1:3) = 0.D0
        DO iat=1, nat
            DO ipol=1, 3
                com(ipol) = com(ipol) + rel_masses(iat)*velocities(istep, iat, ipol) ! / M
            END DO
        END DO
        velocities_(istep, 1, 1:3) = com(1:3)
    END DO
END SUBROUTINE get_COM_velocities



SUBROUTINE calculate_msd_specific_atoms_decompose_d(    &
        positions,                              &
        indices_of_interest,                    &
        msd,                                    &
        stepsize,                               &
        stepsize_inner,                         &
        block_length_dt,                        &
        nr_of_blocks,                           &
        nr_of_t,                                &
        nstep,                                  &
        nat,                                    &
        nat_of_interest                         &
    )
    ! calculates the diffusion coefficient matrix by looping over ipol and jpol
    implicit none

    INTEGER, intent(in)         ::  nstep,  nat,            &   ! Number of timesteps, number of atoms
                                    nat_of_interest,        &   ! number of atoms of interest
                                    nr_of_blocks,           &   ! Nr of blocks to calculate
                                    block_length_dt,        &   ! block length in dt
                                    nr_of_t,                &   ! nr of times of t
                                    stepsize,               &   ! The stepsize, eg 10 to calculate every 10th time
                                    stepsize_inner              ! The inner stepsize

    REAL*8, intent(in),  dimension(nstep, nat, 3)           :: positions
    INTEGER, intent(in), dimension(nat_of_interest)         :: indices_of_interest
    REAL*8, intent(out), dimension(nr_of_blocks, nr_of_t,3,3) :: msd

    ! Local variables:
    INTEGER                     ::  t, tau,                    &        ! Time variables
                                    iat, ipol, jpol,           &        ! Atom count
                                    iat_of_interest,           &        ! Atom count in indices_of interest
                                    iblock                              ! block count
    REAL*8                      ::  msd_this_t !, M

    ! Loop over the block
    
    DO iblock = 1, nr_of_blocks
        ! Loop over time t
        DO t = 1, nr_of_t
            ! Calculate one value of msd for this t in this block 
            ! based on the average over atoms and running window defined by tau
            DO ipol=1,3
                DO jpol=1,3
                    msd_this_t = 0.0d0
                    DO iat_of_interest=1, nat_of_interest
                        iat = indices_of_interest(iat_of_interest)
                        ! Running average achieved by letting tau run through the whole block
                        DO tau = (iblock -1)*block_length_dt+1, iblock*block_length_dt, stepsize_inner
                            msd_this_t = msd_this_t + &
                                (positions(tau+stepsize*t, iat, ipol)-positions(tau, iat, ipol))* &
                                (positions(tau+stepsize*t, iat, jpol)-positions(tau, iat, jpol))
                        END DO
                    END DO
                    msd(iblock, t, ipol, jpol) = msd_this_t
                END DO
            END DO
            
        END DO
    END DO
    msd(:,:,:,:) = msd(:, :,:, :) / DBLE(block_length_dt) / DBLE(nat_of_interest) * DBLE(stepsize_inner)  ! / DBLE(
end SUBROUTINE calculate_msd_specific_atoms_decompose_d

SUBROUTINE calculate_vaf_specific_atoms(        &
        velocities,                             &
        indices_of_interest,                    &
        vaf,                                    &
        stepsize,                               &
        stepsize_tau,                           &
        nr_of_t,                                &
        nr_of_blocks,                           &
        block_length_dt,                        &
        deltaT,                                 &
        integration_method,                     &
        nstep,                                  &
        nat,                                    &
        nat_of_interest,                        &
        vaf_integral                            &
    )

    ! Calcyulates the velocities from the positions and than the autocorrelation function
    ! Averaging over all atoms given by the incices of interest

    implicit none
    INTEGER, intent(in)         ::  nstep,  nat,            &   ! Number of timesteps, number of atoms
                                    nat_of_interest,        &   ! number of atoms of interest
                                    nr_of_t, nr_of_blocks,  &   ! nr of times of t
                                    stepsize,               &   ! The stepsize, eg 10 to calculate every 10th time
                                    stepsize_tau, block_length_dt     ! The stepsize for tau, eg 10 to calculate every 10th time
    CHARACTER*256, intent(in)        ::  integration_method                     
    ! THE ARRAYS:
    REAL*8, intent(in),  dimension(nstep, nat, 3)           :: velocities
    INTEGER, intent(in), dimension(nat_of_interest)         :: indices_of_interest
    REAL*8, intent(out), dimension(nr_of_blocks, nr_of_t)   :: vaf
    REAL*8, intent(out), dimension(nr_of_blocks, nr_of_t)   :: vaf_integral


    ! Local variables:
    INTEGER                     ::  t, tau,            &        ! Time variables
                                    iat, iblock,       &        ! Atom count
                                    iat_of_interest, ipol          ! Atom count in indices_of interest

    REAL*8                      ::  vaf_this_t, vaf_this_t_cumulated !, nr_of_samples_real
    REAL*8, intent(in)          ::  deltaT !, nr_of_samples_real
    INTEGER                     ::  n



    ! I having N blocks:
    DO iblock=1, nr_of_blocks
        ! My time goes from 0 to the time defined as t_end_vaf
        DO t = 0, nr_of_t - 1
            vaf_this_t_cumulated = 0.0d0
            n = 0
            DO iat_of_interest=1, nat_of_interest
                iat = indices_of_interest(iat_of_interest)
                DO tau = (iblock -1)*block_length_dt+1,iblock*block_length_dt, stepsize_tau
                   ! tau = (iblock -1)*block_length_dt+1, 
                    vaf_this_t = 0.0d0
                    DO ipol=1, 3
                        vaf_this_t = vaf_this_t + velocities(tau+stepsize*t, iat, ipol)*velocities(tau, iat, ipol)
                    END DO
                    vaf_this_t_cumulated = ( DBLE(n)*vaf_this_t_cumulated + vaf_this_t ) / DBLE(n+1)
                    n = n+1
                END DO
            END DO

            ! vaf_this_t = vaf_this_t / DBLE(block_length_dt) / DBLE(nat_of_interest) / DBLE(stepsize_tau)

            vaf(iblock, t+1) = vaf_this_t_cumulated 

            ! Using trapezoidal rule to estimate the integral,
            
            SELECT CASE ( integration_method )
                CASE ( 'trapezoid' )
                    IF ( t .eq. 0 ) THEN
                        vaf_integral(iblock, t+1) = 0.5 * deltaT * vaf_this_t_cumulated
                    ELSEIF ( t .eq. ( nr_of_t-1 ) ) THEN
                        vaf_integral(iblock, t+1) = 0.5 * deltaT * vaf_this_t_cumulated  + vaf_integral(iblock, t)
                    ELSE
                        vaf_integral(iblock, t+1) = deltaT * vaf_this_t_cumulated + vaf_integral(iblock, t)
                    ENDIF
                CASE ( 'trapezoid-simple' )
                    IF ( t .eq. 0 ) THEN
                        vaf_integral(iblock, t+1) = 0.0D0
                    ELSE
                        vaf_integral(iblock, t+1) = vaf_integral(iblock, t) + deltaT*0.5*(vaf(iblock, t+1) + vaf(iblock, t))
                    ENDIF

                CASE ( 'simpson' )
                    IF ( t .eq. 0 ) THEN
                        vaf_integral(iblock, t+1) =  deltaT / 3.0D0 * vaf_this_t_cumulated
                    ELSEIF ( t .eq. ( nr_of_t-1 ) ) THEN
                        vaf_integral(iblock, t+1) =  deltaT / 3.0D0 * vaf_this_t_cumulated+vaf_integral(iblock, t)
                    ELSEIF ( MOD(t, 2) .eq. 0 ) THEN
                        vaf_integral(iblock, t+1) =  4.0D0 * deltaT / 3.0D0 * vaf_this_t_cumulated+vaf_integral(iblock, t)
                    ELSEIF ( MOD(t, 2) .eq. 1 ) THEN
                        vaf_integral(iblock, t+1) =  2.0D0 * deltaT / 3.0D0 * vaf_this_t_cumulated+vaf_integral(iblock, t)
                    ENDIF
            
            END SELECT

            ! USING simpson's rule

        END DO
    END DO

END SUBROUTINE calculate_vaf_specific_atoms
