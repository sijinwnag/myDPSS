import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd
from sklearn.linear_model import LinearRegression

class defect_DPSS():
    """
    this object will load the generated data from Yoann's code and do the Linear DPSS methods for the data
    this object assumes that doping is unchanged, just the T is changing

    to do:
    1. fix the vp vn problem
    2. make the code capable for n type material as well
    3. make the code capable for different doping methods
    """

    def __init__(self, path, doping_type='p', plotlifetime=False):
        """
        1. Load the data from the given path
        2. self.doping is a numpy array of doping value corresponding to each lifetime data.
        """
        self.data = pd.read_csv(path)
        variable_type, temp_list, doping_level, excess_dn = self.temp_reader_heading() # gain informaiton about the defect by reading the headings
        self.doping = np.array(list(np.char.split(doping_level, 'cm')))[:, 0].astype(np.float) # this line is to extract the number from the string (number + unit)
        self.plotlifetime = plotlifetime # a boolean input whether you wont to plot the lifetime data as a function of excess carrier concentration
        self.doping_type = doping_type


    def data_extraction(self, row_index=0):
        """
        Take the whole dataframe of defect:
        1. select the defect through row_index
        2. return the lifetime data for this defect under different temperature as a list of array
            lifetime_diff_T: is a list of array for lifetime data under different temprature.
            dn_diff_T: is a lsit of array for excess carrier concentration under different temperatures
            T_unique: is a list of temperature corresponding to each array, is a string of values with units
        3. This function also controls the colour so that it goes from red to blue
        """
        plotlifetime = self.plotlifetime
        # extract the data for one defect:
        defect_data = self.data.iloc[row_index]
        # extract the lifetime data:
        variable_type, temp_list, doping_level, excess_dn = self.temp_reader_heading()
        defect_lifetime = np.array(defect_data)[np.array(variable_type)=='X']
        # split the lifetime data by temperature:
        # define the unique values of T:
        T_unique = np.unique(np.array(temp_list))
        lifetime_diff_T = []
        dn_diff_T = []
        for temperature in T_unique:
            # create an emptylist to collect the data at this temprature:
            lifetime_T = defect_lifetime[np.array(temp_list)==temperature]
            dn_T = np.array(excess_dn)[np.array(temp_list)==temperature]
            # now dn_T is an array of string, convert them into floats
            dn_T = np.array(list(np.char.split(dn_T, 'cm')))[:, 0].astype(np.float)
            lifetime_diff_T.append(lifetime_T)
            dn_diff_T.append(dn_T)
        # now we have a list of list of lifetime data and each list containing the data with the same temperature.
        # plot the data if required:
        if plotlifetime == True:
            fig= plt.figure(facecolor='white', figsize=(5, 5))
            # ax = fig.add_subplot(111)
            # plt.set_size_inches(10.5, 10.5)
            colormap = plt.cm.gist_ncar
            plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(T_unique)))))
            for n in range(len(lifetime_diff_T)):
                plt.plot(dn_diff_T[n], lifetime_diff_T[n], label='T=' +str(T_unique[n]))
            plt.xlabel('$\Delta n$ (cm$^{-3}$)', fontsize=20, fontname="Arial")
            # plt.xlabel(r'Primary T$_{\rm eff}$')
            plt.ylabel('$\u03C4$ (s)', fontsize=20, fontname="Arial")
            # ax.set_aspect("equal")
            # plt.text(0.9, 0.1, '(a)', fontsize=22, transform=ax.transAxes)
            plt.xticks(fontsize=20, fontname="Arial")
            plt.yticks(fontsize=20, fontname="Arial")
            plt.legend(fontsize=15)
            plt.xscale('log')
            plt.savefig('Lifetime plot.png', bbox_inches='tight')
            # plt.title('Lifetime data for a defect')
            plt.show()
        return lifetime_diff_T, dn_diff_T, T_unique


    def temp_reader_heading(self):
        """
        This function takes the loaded dataframe then return four lists:
        variable_type: a list containing string X or y, if it is X that means this column is lifetime data, if it is y this column is the target values
        temp_list: a list of temperature (string, with units) for each column that labeled X
        dopiong_level: a list of doping levels (string, with units) for each column that labeled X
        excess_dn: a list of excess carrier concentration (string, with units) for each column that labeled X.
        """
        # extract the heading.
        headings = list(self.data.columns)
        # extract the information from headings
        # prepare the empty list to collect temperatures, doping levels and excess carrier concentration
        temp_list = []
        doping_level = []
        excess_dn = []
        variable_type = [] # it will be a list of X and y, if it is X means it is a variable if it is y it means that is the target value we want to predict
        for string in headings:
            # if the first element of the string is not a number, then we know it is a part of y rather than lifetime data:
            if string[0].isdigit() == False:
                variable_type.append('y')
            else: # else, we know that it is a lifetime data, read the temprature, doping and dn from the title
                variable_type.append('X')
                temp, doping, dn = string.split('_')
                temp_list.append(temp)
                doping_level.append(doping)
                excess_dn.append(dn)
        return variable_type, temp_list, doping_level, excess_dn


    def linear_fitting(self, plot_linear=True):
        """
        fit the lifetime data for one single defect with linearized form
        if plot==True: then plot the linearized lifetime data
        this function assumes that the doping is constant
        """
        # load data that is saperated by temperature:
        lifetime_diff_T, dn_diff_T, T_unique = self.data_extraction()
        # plot the lineralized SRH equation
        if plot_linear == True:
            fig= plt.figure(facecolor='white', figsize=(5, 5))
            # ax = fig.add_subplot(111)
            colormap = plt.cm.gist_ncar
            plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(T_unique)))))
            for n in range(len(lifetime_diff_T)):
                # print(np.shape(dn_diff_T[n]))
                # print(np.shape(self.doping[:len(dn_diff_T[n])]))
                # print(np.shape(lifetime_diff_T[n]))
                plt.plot((dn_diff_T[n])/(dn_diff_T[n] + self.doping[:len(dn_diff_T[n])]), lifetime_diff_T[n], label='T=' +str(T_unique[n]))
            plt.xlabel('X=' + '$\dfrac{n}{p}$', fontsize=20, fontname="Arial")
            plt.ylabel('$\u03C4$ (s)', fontsize=20, fontname="Arial")
            plt.xticks(fontsize=20, fontname="Arial")
            plt.yticks(fontsize=20, fontname="Arial")
            plt.legend(fontsize=15)
            # ax.set_aspect("equal")
            # plt.text(0.9, 0.1, '(b)', fontsize=22, transform=ax.transAxes)
            # plt.xscale('log')
            # plt.title('Linear Lifetime data for a defect')
            plt.savefig('linear_lifetime.png', bbox_inches='tight')
            plt.show()
        # prepare empty list to collect the slopes and intercept:
        slopelist = []
        interceptlist = []
        # find the thing using linear fitting for different temperatures:
        for n in range(len(lifetime_diff_T)):
            # define the x and y for fitting:
            x_fitting = (dn_diff_T[n])/(dn_diff_T[n] + self.doping[:len(dn_diff_T[n])])
            y_fitting = lifetime_diff_T[n]
            # print(np.shape(x_fitting))
            # print(np.shape(y_fitting))
            # perform the linear fitting
            linear_model = LinearRegression()
            linear_model.fit(x_fitting.reshape(-1, 1), y_fitting)
            intercept = linear_model.intercept_
            interceptlist.append(intercept)
            slope = linear_model.coef_
            slopelist.append(slope)
        return interceptlist, slopelist


    def n1p1SRH(self, Et, T):
        # ni the intrinnsic carrier concentration of silicon
        ni = 5.29e19*(T/300)**2.54*np.exp(-6726/T)
        # the bolzmans constant is always same
        k = 8.617e-5 # boltzmans coonstant, unit is eV/K
        # apply the equation:
        n1 = ni*np.exp(Et/k/T)
        p1 = ni*np.exp(-Et/k/T)
        return n1, p1


    def DPSSplot(self):
        """
        return the k value given the Et and slope and intercept of the linear fitted SHR equation
        each Et will have a k corresponding to it
        this function assumes that doping is unchanged
        this function was assuming vp=vn=1, which is wrong, will figure out later
        """
        # load data that is saperated by temperature:
        lifetime_diff_T, dn_diff_T, T_unique = self.data_extraction()
        # obtain the list of slope and intercept for different temperature:
        interceptlist, slopelist = self.linear_fitting()
        klist_diffT = []
        taup0list_diffT = []
        for n in range(len(slopelist)):
            # simplify the symbol:
            m = slopelist[n]
            h = interceptlist[n]
            # read the temperature of that lifetime data:
            T = float(T_unique[n].split('K')[0])
            # prepare an empty list to collect k and set up Et axis to swing on:
            Etlist = np.linspace(-0.5, 0.21)
            klist = []
            taup0list = []
            # swing across different Et value:
            for Et in Etlist:
                # calculate the p1 and n1:
                n1, p1 = self.n1p1SRH(Et, T)
                # calculate the corresponding k based on the given equation:
                taop0 = ((1 + p1/self.doping[0])*m + p1/self.doping[0]*h)/(1-n1/self.doping[0] + p1/self.doping[0])
                taon0 = m + h - taop0
                # find the thermal velocity:
                vp = 1
                vn = 1
                k = taop0*vp/taon0/vn
                # collect the calculated k
                klist.append(k)
                taup0list.append(taop0)
            # now we have a list of Etlist and Eklist, collect the calculated k into a list of list for different temperature:
            klist_diffT.append(klist)
            taup0list_diffT.append(taup0list)
        # plot the Et vs k under different temperature:
        fig= plt.figure(facecolor='white', figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # print(T_unique)
        counter = 0
        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(T_unique)))))
        for k in klist_diffT:
            # k here is a list
            plt.plot(Etlist, k, label = 'T=' + str(T_unique[counter]))
            counter = counter + 1
        plt.yscale('log')
        plt.xlabel('$E_t-E_i$ (eV)', fontsize=20, fontname="Arial")
        plt.ylabel('k', fontsize=20, fontname="Arial")
        plt.legend(fontsize=15)
        plt.xticks(ticks=[-0.5, -0.25, 0, 0.25, 0.5], fontsize=20, fontname="Arial")
        plt.yticks(fontsize=20, fontname="Arial")
        # ax.set_aspect("equal")
        # plt.text(0.9, 0.1, '(c)', fontsize=22, transform=ax.transAxes)
        plt.savefig('DPSS.png', bbox_inches='tight')
        # plt.title('DPSS analysis plot')
        plt.show()

        # plot the Et vs tao under different temperature:
        # fig0 = plt.figure(facecolor='white', figsize=(5, 5))
        # plt.figure(facecolor='white', figsize=(5, 5))
        fig= plt.figure(facecolor='white', figsize=(5, 5))
        # ax = fig.add_subplot(111)
        # print(T_unique)
        counter = 0
        colormap = plt.cm.gist_ncar
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(T_unique)))))
        for tau in taup0list_diffT:
            # k here is a list
            plt.plot(Etlist, tau, label = '' + str(T_unique[counter].split(' K')[0]))
            counter = counter + 1
        plt.yscale('log')
        plt.xlabel(r'$E_{t}-E_{i}$ (eV)', fontsize=20, fontname="Arial")
        plt.ylabel('$tau$', fontsize=20, fontname="Arial")
        plt.legend(fontsize=15, title = r'T (K)', title_fontsize=15)
        # bbx_to_anchor=(0.5, 1)
        # ncol = 4
        plt.tight_layout()
        plt.subplots_adjust()

        # plt.title('DPSS analysis plot')
        plt.show()
