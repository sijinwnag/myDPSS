from linear_DPSS_obj import *

path = r'C:\Users\sijin wang\Documents\GitHub\Yoann_code\Savedir_example\outputs\2022-02-22-09-39-05_Main_datasetID_0_single_level.csv'

DPSS_obj = defect_DPSS(path, plotlifetime=True)
# lifetime_diff_T, dn_diff_T, T_unique = DPSS_obj.data_extraction()
# DPSS_obj.data_extraction()
# DPSS_obj.linear_fitting
DPSS_obj.DPSSplot()
