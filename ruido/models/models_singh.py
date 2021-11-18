# re-digitized models from Shri Singh's 1995, 1997 papers
# from well logs
import numpy as np


# input: depth below surface (z) and model name
# see the original paper for location of station names
# model returns nan for depths below the well logs
def models_singh(z, model):

    coyo_z_mm = [0, 4, 12.2, 15.3, 20.2, 24.6, 26.3, 29.5, 32.2, 38, 40.1, 45.3, 48, 52.1, 54.1, 56, 57.9, 61, 64.9, 71.3, 73.3, 80., 86.5, 90]
    coyo_b_mm = [13, 16, 32, 6, 40, 32, 27, 32, 47, 32, 40, 64, 56, 64, 48, 69, 38, 42, 32, 45, 73, 56, 69, np.nan]
    coyo_z = (100. / 133.3) * np.array(coyo_z_mm)
    coyo_vs = (800.0 / 63.8) * np.array(coyo_b_mm)

    cu_z = [0, 6, 10, 13, 16, 18, 20, 22, 23, 24, 26, 27, 33, 35, 38, 50, 60, 64, 70, 75]
    cu_vs = [150, 200, 250, 200, 950, 550, 800, 450, 700, 550, 700, 550, 400, 500, 800, 500, 1000, 700, 1000., np.nan]
    cu_z = np.array(cu_z)
    cu_vs = np.array(cu_vs)

    unk_z_mm = [0, 7.1, 33.6, 47.8, 51.6, 69.3, 77., 82.3, 87.8, 93, 100.5, 105.4, 106]
    unk_b_mm = [8, 4, 7, 16, 8, 43, 36, 21, 11, 5, 31, 41, np.nan]
    unk_z = (100. / 133.3) * np.array(unk_z_mm)
    unk_vs = (800.0 / 63.8) * np.array(unk_b_mm)

    tla_z_mm = [0, 10.3, 37.4, 50.8, 58.7, 70.8, 84, 102.4, 105]
    tla_b_mm = [8, 4, 5, 13, 8, 45, 31, 38, np.nan]
    tla_z = (100. / 133.3) * np.array(tla_z_mm)
    tla_vs = (800.0 / 63.8) * np.array(tla_b_mm)

    zar_z_mm = [0, 36.3, 49.8, 53.7, 75.4, 88., 90.9, 102.8, 106.3, 110.3, 115]
    zar_b_mm = [3, 4, 18, 7, 39, 15, 1, 16, 30, 40, np.nan]
    zar_z = (100. / 133.3) * np.array(zar_z_mm)
    zar_vs = (800.0 / 63.8) * np.array(zar_b_mm)

    romac_z_mm = [0, 6.8, 8.6, 17.6, 34.4, 41.2, 45.1, 48.7, 58.3, 59.8, 73.3, 78.2, 83.5, 93.2, 98.4, 107.5, 114.6, 117.2, 124., 130]
    romac_b_mm = [8, 4, 3, 5, 7, 11, 15, 13, 24, 32, 24, 18, 29, 34, 39, 34, 39, 36, 32, np.nan]
    romac_z = (100. / 133.3) * np.array(romac_z_mm)
    romac_vs = (800.0 / 63.8) * np.array(romac_b_mm)

    cdao_z_mm = [0, 7, 10, 21.8, 24.4, 33.2, 37.6, 40, 42., 52.2, 57.4, 64, 71.8, 75]
    cdao_b_mm = [8, 4, 2, 6, 4, 6, 4, 10, 8, 16, 12, 14, 32, np.nan]
    cdao_z = (100. / 133.3) * np.array(cdao_z_mm)
    cdao_vs = (800.0 / 63.8) * np.array(cdao_b_mm)

    assert (len(cu_z) == len(cu_vs))
    assert (len(coyo_z) == len(coyo_vs))
    assert (len(unk_z) == len(unk_vs))
    assert (len(tla_z) == len(tla_vs))
    assert (len(zar_z) == len(zar_vs))
    assert (len(romac_z) == len(romac_vs))
    assert (len(cdao_z) == len(cdao_vs))

    if model == "coyo":
        i = 0
        vs = coyo_vs[0]
        while True:
            if i == len(coyo_z):
                break
            if coyo_z[i] > z:
                break
            vs = coyo_vs[i]
            i += 1

    elif model == "cu":
        i = 0
        vs = cu_vs[0]
        while True:
            if i == len(cu_z):
                break
            if cu_z[i] > z:
                break
            vs = cu_vs[i]
            i += 1
    elif model == "unk":
        i = 0
        vs = unk_vs[0]
        while True:
            if i == len(unk_z):
                break
            if unk_z[i] > z:
                break
            vs = unk_vs[i]
            i += 1
    elif model == "tla":
        i = 0
        vs = tla_vs[0]
        while True:
            if i == len(tla_z):
                break
            if tla_z[i] > z:
                break
            vs = tla_vs[i]
            i += 1
    elif model == "zar":
        i = 0
        vs = zar_vs[0]
        while True:
            if i == len(zar_z):
                break
            if zar_z[i] > z:
                break
            vs = zar_vs[i]
            i += 1
    elif model == "romac":
        i = 0
        vs = romac_vs[0]
        while True:
            if i == len(romac_z):
                break
            if romac_z[i] > z:
                break
            vs = romac_vs[i]
            i += 1

    elif model == "cdao":
        i = 0
        vs = cdao_vs[0]
        while True:
            if i == len(cdao_z):
                break
            if cdao_z[i] > z:
                break
            vs = cdao_vs[i]
            i += 1
    else:
        raise ValueError("Unknown model {}.".format(model))
    return(vs)
