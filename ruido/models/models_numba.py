"""
1-D rule based Earth model for surface wave calculations in Mexico City (basin & hill)

:copyright:
    Laura Ermert (lermert@uw.edu), Nov. 2021
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from numba import jit
import numpy as np
#--------------------------------------------------------------------------------------------------
# - same as models but for use in scripts with numba
#--------------------------------------------------------------------------------------------------
@jit(nopython=True)
def model_cdmx_discrete(z, model, output="v_rho_q", z_is_radius=False):

    model_avail = True
    layered = True
    
    if model[-8:] == "gradient":
        #print("using gradient model instead of layered")
        gradient = True
        layered = False
        model = model[0: 9]
        if model[-1] == "_": model = model[:-1]

    # BEDROCK
    if model in ["cdmx_aovm", "cdmx_cjvm", "cdmx_gmvm", "cdmx_ipvm", "cdmx_mhvm",
                 "cdmx_mpvm", "cdmx_mzvm", "cdmx_ptvm", "cdmx_tlvm", "ESTA", "cdmx_ESTA",
                 "cdmx_mcvm"]:
        clay_depth = 0.0
        sed_depth = 0.
        site_type = "hard"
        print("Hard site")

    # SEDIMENT (NO / VERY LITTLE CLAY)
    elif model in ["cdmx_xcvm", "cdmx_unm", "cdmx_UNM", "UNM", "unm", "TEPE", "cdmx_TEPE", "cdmx_thvm"]:
        clay_depth = 0.0
        sed_depth = 100.0
        site_type = "intermediate"
        print("Intermediate site")

    # SEDIMENT (INCL. CLAY)
    elif model in ["cdmx_bjvm",  "cdmx_test"]:
        clay_depth = 25
        sed_depth = 100.
        site_type = "intermediate"
    elif model == "cdmx_covm":
        clay_depth = 40.0
        sed_depth = 100.
        site_type = "intermediate"
    elif model == "cdmx_apvm":
        clay_depth = 20.0
        sed_depth = 100.0
        site_type = "intermediate"

    elif model == "cdmx_ctvm":
        clay_depth = 55
        sed_depth = 300.
        site_type = "soft"
    elif model == "cdmx_icvm":
        clay_depth = 75.
        sed_depth = 300.
        site_type = "soft"
    elif model == "cdmx_vrvm":
        clay_depth = 50.0
        sed_depth = 300.
        site_type = "soft"
    elif model in ["MULU", "cdmx_MULU"]:
        clay_depth = 30
        sed_depth = 300
        site_type = "soft"
    elif model in ["CIRE", "cdmx_CIRE"]:
        clay_depth = 35
        sed_depth = 300
        site_type = "soft"
    elif model in ["MIXC", "cdmx_MIXC"]:
        clay_depth = 10
        sed_depth = 100
        site_type = "intermediate"

    else:
        model_avail = False

    # model
    if z_is_radius:
        z = 6371000.0 - z # with respect to Earth radius

    if model_avail and layered:

        if z < 0:
            z = 0.
        if z < clay_depth:
            vs = 50.0
            vp = 800.0
            rho = 1250.0
            qs = 60.
            qp = 120.
            fluid_volume_fraction = 0.6 # using porosity as proxy
            # porosity from Ortega Guerrero & Farvolden, 1989
        elif z >= clay_depth and z < sed_depth / 2:
            vs = 400.
            vp = 2500.
            rho = 2000.
            qs = 115.
            qp = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy
        elif z >= sed_depth / 2 and z < sed_depth:
            vs = 800.
            vp = 2500.
            rho = 2000.
            qs = 115.
            qp = 230
            fluid_volume_fraction = 0.2  # using porosity as proxy
        elif z >= sed_depth:
            vs = 1050.
            vp = 2600.
            rho = 2000.
            qs = 115.
            qp = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy
        elif z >= sed_depth + 1000.0:
        # harder bedrock (see cross-section in Singh 95)
            vs = 2100.
            vp = 3600.0
            rho = 2000.0
            qs = 115.
            qp = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy

        if model == "cdmx_test":
            # test the model with less extreme Poisson ratio
            vs *= 2.0

    elif model_avail and gradient:
        d0 = 0
        d1 = clay_depth
        d2 = sed_depth / 2
        d3 = sed_depth
        d4 = 1000.0
        d5 = 2000.0

        z0 = d0
        z1 = (d2 - d1) / 2. + d1
        z2 = (d3 - d2) / 2. + d2
        z3 = (d4 - d3) / 2. + d3
        z4 = (d5 - d4) / 2. + d4
        
        if z < 0:
            z = 0.
        elif z >=0 and z < z1:
            dd = z1 - d0
            vs0 = 50.0 
            vp0 = 800.0 
            rho0 = 1250.0 
            qs0 = 60. 
            qp0 = 120. 
            fluid_volume_fraction0 = 0.6 
            vs1 = 400.
            vp1 = 2500.
            rho1 = 2000.
            qs1 = 115.
            qp1 = 230.
            fluid_volume_fraction1 = 0.2

            vs = vs0 + (vs1 - vs0) / dd * (z - z0)
            vp = vp0 + (vp1 - vp0) / dd * (z - z0)
            qs = qs0 + (qs1 - qs0) / dd * (z - z0)
            qp = qp0 + (qp1 - qp0) / dd * (z - z0)
            rho = rho0 + (rho1 - rho0) / dd * (z - z0)

            fluid_volume_fraction = fluid_volume_fraction0 + (fluid_volume_fraction1 - fluid_volume_fraction0) / dd * (z - z0)
        
            
        elif z >= z1 and z < z2:
            dd = z2 - z1
            vs0 = 400.
            vp0 = 2500.
            rho0 = 2000.
            qs0 = 115.
            qp0 = 230.
            vs1 = 800.
            vp1 = 2500.
            rho1 = 2000.
            qs1 = 115.
            qp1 = 230
            fluid_volume_fraction = 0.2  # using porosity as proxy
            
            vs = vs0 + (vs1 - vs0) / dd * (z - z1)
            vp = vp0 + (vp1 - vp0) / dd * (z - z1)
            qs = qs0 + (qs1 - qs0) / dd * (z - z1)
            qp = qp0 + (qp1 - qp0) / dd * (z - z1)
            rho = rho0 + (rho1 - rho0) / dd * (z - z1)

        elif z >= z2 and z < z3:
            dd = z3 - z2
            vs0 = 800.
            vp0 = 2500.
            rho0 = 2000.
            qs0 = 115.
            qp0 = 230
            vs1 = 1050.
            vp1 = 2600.
            rho1 = 2000.
            qs1 = 115.
            qp1 = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy
            vs = vs0 + (vs1 - vs0) / dd * (z - z2)
            vp = vp0 + (vp1 - vp0) / dd * (z - z2)
            qs = qs0 + (qs1 - qs0) / dd * (z - z2)
            qp = qp0 + (qp1 - qp0) / dd * (z - z2)
            rho = rho0 + (rho1 - rho0) / dd * (z - z2)

        elif z >= z3 and z < z4:
            dd = z4 - z3
            vs0 = 1050.
            vp0 = 2600.
            rho0 = 2000.
            qs0 = 115.
            qp0 = 230.
            vs1 = 2100.
            vp1 = 3600.0
            rho1 = 2000.0
            qs1 = 115.
            qp1 = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy
            vs = vs0 + (vs1 - vs0) / dd * (z - z3)
            vp = vp0 + (vp1 - vp0) / dd * (z - z3)
            qs = qs0 + (qs1 - qs0) / dd * (z - z3)
            qp = qp0 + (qp1 - qp0) / dd * (z - z3)
            rho = rho0 + (rho1 - rho0) / dd * (z - z3)

        elif z >= z4:
        # harder bedrock (see cross-section in Singh 95)
            vs = 2100.
            vp = 3600.0
            rho = 2000.0
            qs = 115.
            qp = 230.
            fluid_volume_fraction = 0.2  # using porosity as proxy

        if model == "cdmx_test":
            # test the model with less extreme Poisson ratio
            vs *= 2.0


    else:
        vp = 0.0
        vs = 0.0
        rho = 0.0
        mu = 0.0
        qp = 0.0
        qs = 0.0
        fluid_volume_fraction = 0.0


    if output == "poroelastic":
        k_s = 35.e9
        k_w  = 2.5e9 
        nu =  0.25
        nu_u = (vp ** 2 - 2. * vs ** 2) / (2. * vp ** 2 - 2. * vs ** 2)
        mu = vs ** 2 * rho
        lam = vp ** 2 * rho - 2 * mu
        k = lam + 2. / 3. * mu  # bulk modulus

        B_clearyrice = (1 / k - 1 / k_s) / (fluid_volume_fraction * (1 / k_w - 1 / k_s) + (1 / k - 1 / k_s))
        # nu_u_clearyrice = (3. * upper_nu + B_clearyrice * (1 - 2. * upper_nu) * (1 - k / k_s)) / (3. - B_clearyrice * (1. - 2. * poisson_ratio) * (1. - k / k_s))
        return(vs, vp, rho, qs, qp, nu, B_clearyrice, nu_u)
    elif output == "elastic":
        #- convert to elastic parameters -----------------------------------------
        eta = 1.0  # isotropic model
        A = C = rho * vp**2
        N = L = rho * vs**2
        F = eta * (A - 2 * L)
        return(rho, A, C, F, L, N, 0., 0.)
    elif output == "v_rho_q":
        return(vs, vp, rho, qs, qp, 0., 0., 0.)
    else:
        return(0., 0., 0., 0., 0., 0., 0., 0.)
