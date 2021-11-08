exp_params = {}


# experiment 1: perfect data
exp_params[1] = {}
exp_params[1]['mu'] = 0.60
exp_params[1]['C'] = 0.65
exp_params[1]['min_instability'] = 0.80
exp_params[1]['max_acceptance_p'] = 0.0
exp_params[1]['random_comp_friction'] = 0.0

# experiment 2: data still follow perfectly the Mohr-Coulomb
# failure criterion, but friction coefficients are distributed
# randomly around the average value given by 'mu'
exp_params[2] = {}
exp_params[2]['mu'] = 0.60
exp_params[2]['C'] = 0.85
exp_params[2]['min_instability'] = 0.80
exp_params[2]['max_acceptance_p'] = 0.0
exp_params[2]['random_comp_friction'] = 0.40 # mu ranges from 0.2 to 1.0

# experiment 3: same as experiment 2, but now some data are allowed
# to violate the Mohr-Coulomb failure criterion
exp_params[3] = {}
exp_params[3]['mu'] = 0.60
exp_params[3]['C'] = 0.65
exp_params[3]['min_instability'] = 0.80
exp_params[3]['max_acceptance_p'] = 0.25
exp_params[3]['random_comp_friction'] = 0.40 # mu ranges from 0.2 to 1.0

# experiment 4: same as experiment 1, but the real coefficient of friction
# is now 0.75, while the computation of the most unstable planes assumes
# mu=0.60
exp_params[4] = {}
exp_params[4]['mu'] = 0.75
exp_params[4]['C'] = 0.65
exp_params[4]['min_instability'] = 0.80
exp_params[4]['max_acceptance_p'] = 0.0
exp_params[4]['random_comp_friction'] = 0.0

# experiment 5: same as experiment 2, but the real coefficient of friction
# is now 0.75, while the computation of the most unstable planes assumes
# mu=0.60
exp_params[5] = {}
exp_params[5]['mu'] = 0.75
exp_params[5]['C'] = 0.65
exp_params[5]['min_instability'] = 0.80
exp_params[5]['max_acceptance_p'] = 0.0
exp_params[5]['random_comp_friction'] = 0.40 # mu ranges from 0.2 to 1.0

# experiment 6: same as experiment 3, but the real coefficient of friction
# is now 0.75, while the computation of the most unstable planes assumes
# mu=0.60
exp_params[6] = {}
exp_params[6]['mu'] = 0.75
exp_params[6]['C'] = 0.65
exp_params[6]['min_instability'] = 0.80
exp_params[6]['max_acceptance_p'] = 0.25
exp_params[6]['random_comp_friction'] = 0.40 # mu ranges from 0.2 to 1.0


