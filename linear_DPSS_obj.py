import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import pandas as pd

class Linear_DPSS():
    """
    this object will load the generated data from Yoann's code and do the Linear DPSS methods for the data
    """

    def __init__(self, path, doping_type='p', plotlifetime=False, row_index):
        """
        1. Load the data from the given path
        """
        self.data = pd.read_csv(path)
        variable_type, temp_list, doping_level, excess_dn = self.temp_reader_heading() # gain informaiton about the defect by reading the headings
        self.doping = doping_level
        self.plotlifetime = plotlifetime # a boolean input whether you wont to plot the lifetime data as a function of excess carrier concentration
        self.doping_type = doping_type


    def data_extraction(self, row_index, plotlifetime=False):
        """
        Take the whole dataframe of defect:
        1. select the defect through row_index
        2. return the lifetime data for this defect under different temperature as a list of array
            lifetime_diff_T: is a list of array for lifetime data under different temprature.
            dn_diff_T: is a lsit of array for excess carrier concentration under different temperatures
            T_unique: is a list of temperature corresponding to each array
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
            plt.figure()
            for n in range(len(lifetime_diff_T)):
                plt.plot(dn_diff_T[n], lifetime_diff_T[n], label='T=' +str(T_unique[n]))
            plt.xlabel('excess carrier concentration ($cm^{-3}$)')
            plt.ylabel('lifetime (s)')
            plt.legend()
            plt.xscale('log')
            plt.title('Lifetime data for a defect under different temperatures')
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


    def linear_fitting(self, defect_lifetimedata, plot_linear=False):
        """
        fit the lifetime data for one single defect with linearized form
        if plot==True: then plot the linearized lifetime data
        """
        # load data that is saperated by temperature:
        lifetime_diff_T, dn_diff_T, T_unique = data_extraction(self, row_index)
        if plot_linear == True:
            plt.figure()
            for n in range(len(lifetime_diff_T)):
                plt.plot(dn_diff_T[n], self.doping_level + lifetime_diff_T[n], label='T=' +str(T_unique[n]))
            plt.xlabel('excess carrier concentration ($cm^{-3}$)')
            plt.ylabel('lifetime (s)')
            plt.legend()
            plt.xscale('log')
            plt.title('Lifetime data for a defect under different temperatures')
            plt.show()
