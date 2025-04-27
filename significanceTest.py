import scipy.stats as stats
import numpy as np
base_set = np.array([0.7386, 0.7522, 0.7484, 0.7454, 0.7492, 
                    0.7462, 0.7446, 0.7482, 0.7502, 0.7468,
                    0.7458])
twc_mdbl_rosh_allhyp = np.array([0.7562, 0.7502, 0.7532, 0.753, 0.7524,
                             0.7536, 0.7524, 0.752, 0.7474, 0.7572,
                             0.7548])

twc_mdbl_allhyp = np.array([0.7504, 0.7476, 0.7516, 0.7540, 0.7516,
                             0.7562, 0.7490, 0.7458, 0.7504, 0.7518,
                             0.7556])

t_stat_twc_mdbl_rosh_allhyp, p_value_twc_mdbl_rosh_allhyp = stats.ttest_ind(base_set, twc_mdbl_rosh_allhyp, equal_var=False)
print(f"t-statistic_rosh: {t_stat_twc_mdbl_rosh_allhyp}, p-value_rosh: {p_value_twc_mdbl_rosh_allhyp}")

median_base_set = np.median(base_set)
median_twc_mdbl_rosh_allhyp = np.median(twc_mdbl_rosh_allhyp)
print(f"Median base set: {median_base_set}, Median twc_mdbl_rosh_allhyp: {median_twc_mdbl_rosh_allhyp}")

# mean_twc_mdbl_allhyp, std_twc_mdbl_allhyp = np.mean(twc_mdbl_allhyp), np.std(twc_mdbl_allhyp)
# print(f"Mean twc_mdbl_allhyp: {mean_twc_mdbl_allhyp}, Std twc_mdbl_allhyp: {std_twc_mdbl_allhyp}")

t_stat_twc_mdbl_allhyp, p_value_twc_mdbl_allhyp = stats.ttest_ind(base_set, twc_mdbl_allhyp, equal_var=False)
median_twc_mdbl_allhyp = np.median(twc_mdbl_allhyp)
print(f"t-statistic: {t_stat_twc_mdbl_allhyp}, p-value: {p_value_twc_mdbl_allhyp}")
print(f"Median base set: {median_base_set}, Median twc_mdbl_allhyp: {median_twc_mdbl_allhyp}")