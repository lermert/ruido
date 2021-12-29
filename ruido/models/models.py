"""
1-D rule based Earth model for surface wave calculations in Mexico City (basin & hill)

:copyright:
    Laura Ermert (lermert@uw.edu), Nov. 2021
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""
#--------------------------------------------------------------------------------------------------
#- Ciudad de Mexico, approximate model for the sediment & upper part of volcanics
#- based on info from Perez Cruz 1988, Singh et al. 1995, Singh et al. 1997, Shapiro et al. 2002
#- plus educated guesses to fill the gaps between ~ 100 m and the sediment bottom
#- model names refer to locations of the RSVM seismic network and the Geoscope station G.UNM
#  Additional data from:
# - Aquitard thickness prescribing to lake (> 0 m) or hill (=0 m) designation,
#   and to upper_z value of clay thickness: Solano Rojas et al., 2015
#   http://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-33222015000200011
# - Approximate sediment depth: Information from well logs, Dario Solano Rojas & Diana Morales, pers. comm.
# - Approximate shear wave velocity, shallow: Singh et al., 1997
# - Approximate P wave velocity and density: Singh et al., 1995, Singh et al. 1997, Perez Cruz 1988
# - Fluid volume fraction: Ortega Guerrero & Farvolden, 1989
# - Bulk modulus of basalt grains:
# - Bulk modulus of water:
# 
#  !! Note: "lake" vs. "hill" in this model only indicates the presence or absense of the aquitard,
#  and is not exactly following geotechnical classification
# - same as models_numba but for use in scripts without numba

#--------------------------------------------------------------------------------------------------
def model_cdmx_discrete(z, model, output="v_rho_q", z_is_radius=False):
    
    # BEDROCK
    if model in ["cdmx_ipvm", "cdmx_aovm", "cdmx_gmvm", "cdmx_cjvm",
    "cdmx_mpvm", "cdmx_mzvm", "cdmx_ptvm", "cdmx_tlvm", "ESTA",
    "cdmx_mcvm"]:
        sed_depth=0.
        zone="hill"

    # SEDIMENT (NO / VERY LITTLE CLAY)
    elif model == "cdmx_xcvm":
        zone = "hill"
        upper_z = 0.0
        sed_depth = 15.0
    elif model in ["cdmx_thvm", "TEPE"]:
        zone="hill"
        sed_depth = 50.0
    elif model in ["cdmx_unm", "cdmx_mhvm"]:
        sed_depth = 100.0
        zone = "hill"

    # SEDIMENT (INCL. CLAY)
    elif model in ["cdmx_bjvm",  "cdmx_test"]:
        sed_depth = 200.
        zone = "lake"
        upper_z = 25
    elif model == "cdmx_icvm":
        sed_depth = 400.
        zone = "lake"
        upper_z = 75.
    elif model == "cdmx_covm":
        sed_depth = 300.
        zone = "lake"
        upper_z = 40.
    elif model == "cdmx_ctvm":
        zone = "lake"
        upper_z = 55
        sed_depth = 300.
    elif model == "cdmx_vrvm":
        zone = "lake"
        upper_z = 50.0
        sed_depth = 300.
    elif model == "cdmx_apvm":
        zone = "lake"
        upper_z = 20.0
        sed_depth = 200.0
    elif model == "MULU":
        zone = "lake"
        upper_z = 30
        sed_depth = 300
    elif model == "CIRE":
        zone = "lake"
        upper_z = 35
        sed_depth = 300
    elif model == "MIXC":
        zone = "lake"
        upper_z = 10
        sed_depth = 300
    else:
        raise ValueError("Unknown model {}".format(model))

    # model
    if z_is_radius:
        z = 6371000.0 - z # with respect to Earth radius

    if zone == "lake":
        upper_z = upper_z
        upper_vs = 50.
        upper_vp = 800.
        upper_rho = 1250.
        fluid_volume_fraction = 0.6 # using porosity as proxy
        # porosity from Ortega Guerrero & Farvolden, 1989

    if zone == "hill":
        upper_z = 0.
        upper_vs = 400.
        upper_vp = 2500.
        upper_rho = 2000.0
        fluid_volume_fraction = 0.2  # using porosity as proxy
        # porosity from Ortega Guerrero & Farvolden, 1989
    if z < 0:
        z = 0.
    if z < upper_z:
        vs = upper_vs
        vp = upper_vp
        rho = upper_rho
        qs = 60.
        qp = 120.
    elif z >= upper_z and z < sed_depth / 2:
        vs = 400.
        vp = 2500.
        rho = 2000.
        qs = 115.
        qp = 230.
    elif z >= sed_depth / 2 and z < sed_depth:
        vs = 800.
        vp = 2500.
        rho = 2000.
        qs = 115.
        qp = 230
    elif z >= sed_depth:
        vs = 1050.
        vp = 2600.
        rho = 2000.
        qs = 115.
        qp = 230.
    elif z >= sed_depth + 1000.0:
    # harder bedrock (see cross-section in Singh 95)
        vs = 2100.
        vp = 3600.0
        rho = 2000.0
        qs = 115.
        qp = 230.
    if zone == "lake":
        if z >= upper_z and z < sed_depth / 4:
            vs = 200.
            vp = 1550.
            rho = 2000.
            qs = 60.
            qp = 120.

    if model == "cdmx_test":
        # test the model with less extreme Poisson ratio
        vs *= 2.0

    if output == "poroelastic":
        k_s = 35.e9
        k_w  = 2.5e9 
        poisson_ratio = (vp ** 2 - 2. * vs ** 2) / (2. * vp ** 2 - 2. * vs ** 2)
        upper_nu = (upper_vp ** 2 - 2. * upper_vs ** 2) / (2. * upper_vp ** 2 - 2. * upper_vs ** 2)
        mu = upper_vs ** 2 * upper_rho
        lam = upper_vp ** 2 * upper_rho - 2 * mu
        k = lam + 2. / 3. * mu  # bulk modulus
        B_clearyrice = (1 / k - 1 / k_s) / (fluid_volume_fraction * (1 / k_w - 1 / k_s) + (1 / k - 1 / k_s))
        nu_u_clearyrice = (3. * upper_nu + B_clearyrice * (1 - 2. * upper_nu) * (1 - k / k_s)) / (3. - B_clearyrice * (1. - 2. * poisson_ratio) * (1. - k / k_s))
        return(vs, vp, rho, qs, qp, poisson_ratio, B_clearyrice, nu_u_clearyrice)
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
