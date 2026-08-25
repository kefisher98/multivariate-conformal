
##################################################################################################
# USER: FILL IN CLUSTER COMMANDS AND SETTINGS 
################################################################################################## 
task_id=1
task_count=1




##################################################################################################
# construct stress covariance matrix
##################################################################################################

blocks=10
cov_type=stress
symmetric=true
SOR_approx=false

julia src/precompute_covariance_blockwise.jl $task_id \
                                             $task_count \
                                             $blocks \
                                             $cov_type \
                                             $symmetric \
                                             $SOR_approx

blocks=10
cov_type=stress_energy
symmetric=false
SOR_approx=false

julia src/precompute_covariance_blockwise.jl $task_id \
                                             $task_count \
                                             $blocks \
                                             $cov_type \
                                             $symmetric \
                                             $SOR_approx



julia src/precompute_energy_stress_covariance.jl

rm data/covariance/stress_energy_compiled_K.csv

##################################################################################################
# construct force covariance matrix
##################################################################################################

blocks=10
cov_type=force
symmetric=true
SOR_approx=true

julia src/precompute_covariance_blockwise.jl $task_id \
                                             $task_count \
                                             $blocks \
                                             $cov_type \
                                             $symmetric \
                                             $SOR_approx

blocks=10
cov_type=force_energy
symmetric=false
SOR_approx=false

julia src/precompute_covariance_blockwise.jl $task_id \
                                             $task_count \
                                             $blocks \
                                             $cov_type \
                                             $symmetric \
                                             $SOR_approx


julia src/precompute_sor_approximation.jl


zip data/covariance/force_SOR_30_200.zip data/covariance/force_SOR_30_200.csv
rm data/covariance/force_energy_compiled_K.csv 

