#for j in 200 300 50; do
#    sed -i -e 's/\(dt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$j' /' params.py
#    python py_mpwp_v0_2.py --out ktpwpkdt_dt"$j"_we
#done

sed -i -e 's/\(dt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '100' /' params.py
for k in 0.1 0.2; do 
    sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$k' /' params.py
    python py_mpwp_v0_2.py --out ktpwpkdt_snow"$k"_we
done

sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.0' /' params.py

for i in 1.0 2.0 5.0; do
    sed -i -e 's/\(dz[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
    python py_mpwp_v0_2.py --out ktpwpkdt_dz"$i"_we
done
sed -i -e 's/\(dz[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '3.0' /' params.py
