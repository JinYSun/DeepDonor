dataset=SM

# Basis set.
basis_set=6-31G

# Grid field.
radius=0.75
grid_inter=0.3
grid_interval=0.3
python preprocess.py $dataset $basis_set $radius  $grid_inter

