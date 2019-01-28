for i in 0.5 2.0 3.0 1.0; do
    sed -i -e 's/\(dz[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
    python py_mpwp_v0_2_hamfist.py --out ktall_dz_$i
    echo "$i"
done

for i in 0.0005 0.001 0.00005 0.00001 ; do
    sed -i -e 's/\(OR_timescale[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
    python py_mpwp_v0_2_hamfist.py --out ktall_OR_$i
    echo "$i"
done

sed -i -e 's/\(OR_timescale[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.0001' /' params.py

echo "returned to norm"

for i in 0.15 0.1 0.05; do
    sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
    python py_mpwp_v0_2_hamfist.py --out ktall_snow_$i
    echo "$i"
done

sed -i -e 's/\(h_snow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.2' /' params.py

echo "done"

