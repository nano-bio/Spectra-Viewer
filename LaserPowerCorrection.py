from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as inter


h = 6.626070040E-34
cspeed = 299792458
eV = 1.60218e-19
power_meter_relative_error = 0.05
power_meter_averaging_number = 20


def refrectionindex_air(lam):  # Input in nm from E. R. Peck and K. Reeder. Dispersion of air, J. Opt. Soc. Am. 62, 958-962 (1972)
    lam = lam/1000               # Valid from 185 - 1700 nm @ 15°C 101.325 kPa 450ppm CO_2
    A = 8.06051e-5
    B = 2.480990e-2
    C = 132.274
    D = 1.74557e-4
    E = 39.32957
    return (A+B/(C-lam**(-2))+D/(E-lam**(-2)))+1


def frequency_air(lam):  # Input in nm output in Hz
    return (cspeed/refrectionindex_air(lam))/(lam/(1e9))


def wavenumber(lam):  # Input in nm, Output in cm^(-1)
    return (frequency_air(lam)/cspeed)/100


def energy(lam):  # Input in nm s, output in eV
    return h*frequency_air(lam)/eV


class Power(object):
    power_meter_relative_error = 0.05
    power_meter_averaging_number = 20

    def __init__(self, wls_data, data_cube,  data_cube_err, powerfilename):

        self.power0 = np.loadtxt(powerfilename, delimiter=',', skiprows=2)

        self.ex = self.power0[:, 0]
        self.ey = self.power0[:, 1]
        self.ey_error = abs((self.power0[:, 1] * power_meter_relative_error)/np.sqrt(power_meter_averaging_number))
        xi = np.argsort(wls_data)
        self.csx = wls_data[xi]
        self.csy = data_cube[xi,:]
        self.csy_error = data_cube_err[xi,:]

        self.csy[self.csy == 0] = 1e-5
        self.raw = self.csy
        self.csy = -np.log(self.csy)

        self.pt = inter.PchipInterpolator(np.sort(self.csx), self.csy,axis=0)
        self.pt_error = inter.PchipInterpolator(np.sort(self.csx), self.csy_error,axis=0)

        self.eyy = np.array((self.ey*self.ex)/(h*cspeed))
        self.eyy_error = np.array(abs(self.ex/(h*cspeed)*self.ey_error))
        self.eyy = np.around(self.eyy, 0)

        self.eyyy = self.eyy
        self.exxx = self.ex

        self.noise = 0
        self.exx = self.ex[np.where(self.eyy > self.noise)[0]]
        self.eyy = self.eyy[self.eyy > self.noise]

        self.ptt = inter.PchipInterpolator(self.exx, self.eyy)
        self.ptt_error = inter.PchipInterpolator(self.exx, self.eyy_error)

        self.depletion = self.pt(self.csx)/(self.ptt(self.csx))[:,None]*1e26
        self.depletion_error = abs(-1./(self.raw*self.ptt(self.csx)[:,None]))*self.csy_error + abs(-np.log(self.raw)/(self.ptt(self.csx)[:,None])**2)*self.ptt_error(self.csx)[:,None]

        self.csf = frequency_air(self.csx)
        self.csw = wavenumber(self.csx)
        self.cse = energy(self.csx)