
##################################################################################################
# remove existing files
##################################################################################################

if test -f data/force_energy/cr1_scaled.csv ; then
  rm data/force_energy/cr1_scaled.csv
fi
if test -f data/force_energy/crq_scaled.csv ; then
  rm data/force_energy/crq_scaled.csv
fi
if test -f data/force_energy/cw_scaled.csv ; then
  rm data/force_energy/cw_scaled.csv
fi
if test -f data/force_energy/mv_scaled.csv ; then
  rm data/force_energy/mv_scaled.csv
fi



##################################################################################################
# calibrate energy and force
##################################################################################################
calibrate=200

unzip data/covariance/force_SOR_30_200.zip -d data/covariance/

julia src/energy_force.jl $calibrate

rm data/covariance/force_SOR_30_200.csv

julia src/make_energy_force_plots.jl
