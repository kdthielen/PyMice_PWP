python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out  year1_ref_$i
for i in 0.2 0.3 0.4 0.6 0.7 0.8 ; do
	sed -i -e 's/\(Div_yr[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_full.npz --init 336_prof.npz --out  ek_2
done

sed -i -e 's/\(Div_yr[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.5' /' params_v3.py


for i in 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9 1.0; do
	sed -i -e 's/\(h_i0[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out  year1_hi0_$i
done

sed -i -e 's/\(h_i0[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.5' /' params_v3.py

for i in 0.5 0.6 0.7 0.8 0.95; do
	sed -i -e 's/\(A_0[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out year1_A0_$i
done

sed -i -e 's/\(A_0[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.9' /' params_v3.py


for i in 0.001 0.00125 0.00175 0.002; do 
	sed -i -e 's/\(cd_air[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out year1_cda_$i
done

sed -i -e 's/\(cd_air[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.0015' /' params_v3.py


for i in  0.0 0.000001 0.0000012 0.0000014 0.0000016 0.0000018 0.000002; do 
	sed -i -e 's/\(ekman[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out year1_ekman_$i
done

sed -i -e 's/\(ekman[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.0000014' /' params_v3.py

for i in 0.0 0.05 0.15 0.2 0.25 0.3; do 
	sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out year1_hs_$i
done

sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.1' /' params_v3.py

for i in 0.9 0.7 0.6 0.5; do 
	sed -i -e 's/\(R_b[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params_v3.py
	python py_mpwp_v0_3.py --force era5-soccom_year.npz --init 336_prof.npz --out year1_rb_$i
done

sed -i -e 's/\(R_b[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.8' /' params_v3.py



