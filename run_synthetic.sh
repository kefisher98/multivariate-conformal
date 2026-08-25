

##################################################################################################
# Synthetic propagation study
##################################################################################################
rho_test=0.3
rho_calibrate=0.3
julia src/synthetic.jl $rho_test $rho_calibrate

rho_test=0.9
rho_calibrate=0.9
julia src/synthetic.jl $rho_test $rho_calibrate

rho_test=0.99
rho_calibrate=0.0
julia src/synthetic.jl $rho_test $rho_calibrate


julia src/make_synthetic_plots.jl
