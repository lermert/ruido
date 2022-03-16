"""
1-D rule based Earth model for surface wave calculations in Mexico City (basin & hill)

:copyright:
    Laura Ermert (lermert@uw.edu), Nov. 2021
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
from numba import jit

#--------------------------------------------------------------------------------------------------
# - same as models but for use in scripts with numba
#--------------------------------------------------------------------------------------------------
@jit(nopython=True)
def model_cdmx_discrete(z, model, output="v_rho_q", z_is_radius=False):
   
    # BEDROCK
    if model in ["cdmx_aovm", "cdmx_cjvm", "cdmx_gmvm", "cdmx_ipvm", "cdmx_mhvm",
    "cdmx_mpvm", "cdmx_mzvm", "cdmx_ptvm", "cdmx_tlvm", "ESTA", "cdmx_ESTA",
    "cdmx_mcvm"]:
        clay_depth = 0.0
        sed_depth = 0.
        site_type = "hard"

    # SEDIMENT (NO / VERY LITTLE CLAY)
    elif model in ["cdmx_xcvm", "cdmx_unm", "cdmx_UNM", "UNM", "unm", "TEPE", "cdmx_TEPE", "cdmx_thvm"]:
        clay_depth = 0.0
        sed_depth = 100.0
        site_type = "intermediate"

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

    # model
    if z_is_radius:
        z = 6371000.0 - z # with respect to Earth radius

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
