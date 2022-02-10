# %%-- Import
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import semiconductor
# from scipy.optimize import fsolve
# %%-

#%%-- define functions
# define the law of mass action
def n_lawofmass(p, T):
    ni = 5.29e19*(T/300)**2.54*np.exp(-6726/T) # intrinsic carrier concentration of silicon at 300K cm-3
    n = ni**2/p0
    return n


# the SRH densities
def n1p1SRH(Et, T):
    # Nc: the effective density of states of silicon in conduction band
    Nv = 3.5e15*T**(3/2)
    Nc = 6.2e15*T**(3/2)
    # the bolzmans constant is always same
    k = 8.617e-5 # boltzmans coonstant, unit is eV/K
    n1 = Nc*np.exp(-(1.12-Et)/k/T)
    p1 = Nv*np.exp(-(Et-0)/k/T)
    return [n1, p1]


# define the lifetime function for SRH recombination
def SRHlifetime(dn, p0, p1, n0, n1, taon0, taop0):
    taoSRH = taon0*(p0 + p1 + dn)/(n0 + p0 + dn) + taop0*(n0 + n1 + dn)/(
        n0 + p0 + dn)
    return taoSRH


# defien the Auger recombination lifetime
def Augerlifetime(n, dn):
    Cn = 2.8e-31
    Cp = 0.99e-31
    taoAuger = 1/(Cn*n**2 + Cp*n*dn)
    return taoAuger


# define the radiative recombination
def Radlifetime(p0, dn, T):
    # use law of mass action to calculate n0
    n0 = n_lawofmass(p0, T)
    # define B
    B = 9.5e-15
    radtao = 1/B/(n0 + p0 + dn)
    return radtao


# define the SRH density using intrinsic carrier density
def n1p1SRH2(Et_Ei, T):
    # the Et in this equation is the Et-Ei instead of the Et-Ev
    ni = 5.29e19*(T/300)**2.54*np.exp(-6726/T) # intrinsic carrier
    # concentration of silicon addapted from PV education
    k = 8.617e-5 # boltzmans coonstant, unit is eV/K
    n1 = ni*np.exp(Et_Ei/k/T)
    p1 = ni*np.exp(-Et_Ei/k/T)
    return [n1, p1]

# define the capture time constant
def taoptaon(sigmap, sigman, Nt, vp, vn):
    # calculate time constant
    taup = 1/sigmap/vp/Nt
    taun = 1/sigman/vn/Nt
    return [taup, taun]


def DPSSSRHtau(Et_Ei, p0, n0, dn, k, taup0, T):
    # Et is the energy level of the defect relative to intrinsic fermi energy
    # p0 and n0 are the carrier densities at equilibrium after doping
    # dn is the excess carrier concentration
    # k is the ratio taun/taup
    # taup is the capture time constant for holes
    # calculate p1 and n1
    [n1, p1] = n1p1SRH2(Et_Ei, T)
    # calculate thermal velocity
    vp = np.sqrt(3*1.38e-23*T/(0.39*9.11e-31))
    vn = np.sqrt(3*1.38e-23*T/(0.26*9.11e-31))
    numerator = taup0*(p0 + p1 + dn)/k*(vp/vn) + taup0*(n0 + n1 + dn)
    denominator = p0 + n0 + dn
    return numerator/denominator


def Green_1990(vals, temp, Egratio, **kargs):
    """
     This form as described by Green in 10.1063/1.345414.
     inputs:
        vals: (dic)
            the effect mass values
        temp: (float)
            the temperature in kelvin
    outputs:
        vel_th_c: (float)
            the termal velocity for the conduction in cm/s
        vel_th_v: (float)
            the termal velocity for the valance band in cm/s
    """

    # the values relative to the rest mass
    ml = vals['ml'] * const.m_e

    mt = vals['mt'] * Egratio * const.m_e

    delta = np.sqrt((ml - mt) / ml)
    # conduction band effective mass
    mth_c = 4. * ml / (
        1. + np.sqrt(ml / mt) * np.arcsin(delta) / delta)**2

    vel_th_c = np.sqrt(8 * const.k * temp / np.pi / mth_c)
    # valance band effective mass, its a 7 order poynomial fit
    mth_v = np.sum(
        [vals['meth_v' + str(i)] * temp**i for i in range(8)]) * const.m_e

    vel_th_v = np.sqrt(8 * const.k * temp / np.pi / mth_v)

    # adjust the values from m/s to cm/s and return
    return vel_th_c * 100, vel_th_v * 100


def taop0ktaon0_liearDPSS(p1, n1, p0, slope, intercept, vn, vp):
    taop0 = ((1 + p1/p0)*slope + p1/p0*intercept)/(1-n1/p0 + p1/p0)
    taon0 = slope + intercept - taop0
    k = taop0*vp/taon0/vn
    return [taop0, taon0, k]


def SRHlifetimegenerator(Et_Ei, vn, vp, dn, Nt, sigman, sigmap, T, p0):
    [n1, p1] = n1p1SRH2(Et_Ei, T)
    n0 = n_lawofmass(p0, T)
    [taop0, taon0] = taoptaon(sigmap, sigman, Nt, vp, vn)
    SRHtal = SRHlifetime(dn, p0, p1, n0, n1, taon0, taop0)
    return SRHtal
# %%-


#%%-- define the parameters
# set up the x axis
dn = 1e15
# print(dn)
# set the given parameters
Et_Ei = np.linspace(-0.365, 0.33) # eV
me = 0.26*9.11e-31
mp = 0.39*9.11e-31
Nt = 1e12
sigman = 2.4e-14
sigmap = 0.8e-14
T = 300 # K
vn = np.sqrt(3*1.38e-23*T/me)*100 # cm/s
vp = np.sqrt(3*1.38e-23*T/mp)*100 # cm/s
realk = sigman/sigmap
# calculate the
# define the parameters
# p0 is the doping density
p0 = 1e16 # cm-3
# %%-

#%%-- reproduce Yanpaper DPSS Figure 4.1 a
# start plotting
plt.figure()
legendlist = []
for temp in [300]:
    # calculate the SRH densities
    [n1, p1] = n1p1SRH2(Et_Ei, temp) # the temperature is assumed to be 300K
    # calculate the intrinsic carrier concentration
    n0 = n_lawofmass(p0, temp)
    # calculate taun and taup
    [taop0, taon0] = taoptaon(sigmap, sigman, Nt, vp, vn)
    # calcualte the SRH lifetime
    SRHtal = SRHlifetime(dn, p0, p1, n0, n1, taon0, taop0)
    plt.plot(dn, SRHtal)
    legendtext = 'T=' + str(temp) + 'K'
    legendlist.append(legendtext)
plt.yscale("log")
plt.xscale("log")
plt.xlabel('Excess Carrier Density cm-3')
plt.ylabel('Carriers Lifetime(s)')
plt.title('Et-Ei=' + str(Et_Ei))
plt.legend(legendlist)
# plt.show()
# try to fit using DPSS

# for each Et find the optimum k and taop0
# the fixed parameters are p0, n0, dn, T
# convert the data into the linear form
n = dn + n0
p = dn + p0
X = n/p
# print(n)
# print(X)
# print(p)
plt.figure()
plt.plot(X, SRHtal)
plt.xlabel('Linear parameter X=n/p')
plt.ylabel('SRH lifetime (s)')
plt.title('linear SRH plot')
# plt.show()

# linear fit the data
xdata = X
ydata = SRHtal
z = np.polyfit(xdata, ydata, 1)
# print('The linear fit slope and intersection are: ', str(z))
slope = z[0]
yintersept = z[1] # unit of s
# calculate the thermal velocity ratio
# check if the slope is expected
slopetheory = taop0 - n1*taop0/p0 - p1*taon0/p0
intersecttheory = taon0 + taop0*n1/p0 + p1*taon0/p0
# print('The expected slope and intersection are: ', str(slopetheory), str(intersecttheory))
# define the mt and ml
val = {
  "ml": 0.9163,
  "mt": 0.1905
}
# calculate the thermal velocities (simplified)
# print(vel_th_e, v_th_h)
# solve the system of equation assuming different Et
# Etlist = [-0.33]
Etlist = [-0.33]
DPSSk = []
DPSStaop = []
for Et_Ei in Etlist:
    # calculate the n1 p1, which depend on the Ei
    [n1, p1] = n1p1SRH2(Et_Ei, T)
    # solve system of linear equation to get  and k
    # let x[0] be k and x[1] be the taop
    [taop0fit, taon0fit, kfit] = taop0ktaon0_liearDPSS(p1, n1, p0, slope, yintersept, vn, vp)
    # compare the real k with the fitted k
    print('The fitted k and taop given the correct Et are: ', kfit, taop0fit)
    print('The real k and taop0 are: ', realk, taop0)

# try this for swipping Et
Etlist = np.arange(-0.5, 0.5, 0.01)
DPSSk = []
DPSStaop = []
for Et_Ei in Etlist:
    # calculate the n1 p1, which depend on the Ei
    [n1, p1] = n1p1SRH2(Et_Ei, T)
    # solve system of linear equation to get  and k
    # let x[0] be k and x[1] be the taop
    [taop0fit, taon0fit, kfit] = taop0ktaon0_liearDPSS(p1, n1, p0, slope, yintersept, vn, vp)
    # compare the real k with the fitted k
    DPSSk.append(kfit)
    DPSStaop.append(taop0fit)

plt.figure()
plt.yscale("log")
plt.plot(Etlist, DPSSk)
plt.title('DPSSk vs Et')
plt.xlabel('Et relative to Ei (eV)')
plt.ylabel('k')
# plt.show()

plt.figure()
plt.yscale("log")
plt.plot(Etlist, DPSStaop)
plt.title('DPSStaop vs Et')
plt.xlabel('Et relative to Ei (eV)')
plt.ylabel('s')
plt.show()
# %%-

# %%-- Plot the Et vs Lifetime.
plt.figure()
# calculate the SRH densities
[n1, p1] = n1p1SRH2(Et_Ei, temp) # the temperature is assumed to be 300K
# calculate the intrinsic carrier concentration
n0 = n_lawofmass(p0, temp)
# calculate taun and taup
[taop0, taon0] = taoptaon(sigmap, sigman, Nt, vp, vn)
# calcualte the SRH lifetime
SRHtal = SRHlifetime(dn, p0, p1, n0, n1, taon0, taop0)
SRHtal = np.log10(SRHtal)
plt.plot(Et_Ei, SRHtal)
plt.xlabel('$E_i$')
plt.ylabel('Log of Carriers Lifetime(s)')
plt.legend(legendlist)
plt.title('log of Lifetime vs defect energy')
plt.show()
# %%-
