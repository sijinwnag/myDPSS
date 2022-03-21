# %%-- Imports and set ups
from linear_DPSS_obj import *
path = r'C:\Users\sijin wang\Documents\GitHub\SRH_sklearn_playwithdata\One_single_Level_defect\lifetime_dataset_example.csv'
DPSS_obj = defect_DPSS(path, plotlifetime=True)
# %%-

# %%-- Perform tasks.
# lifetime_diff_T, dn_diff_T, T_unique = DPSS_obj.data_extraction()
# DPSS_obj.data_extraction()
# DPSS_obj.linear_fitting
DPSS_obj.DPSSplot()
# %%-
