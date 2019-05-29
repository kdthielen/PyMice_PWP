
for i in  0.00 0.20 0.5 0.7 0.89; do
	sed -i -e 's/\(A_grow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
	sed -i -e 's/\(A_melt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i' /' params.py
	python py_mpwp_v0_2.py --out chloe_bc_A_$i
done

sed -i -e 's/\(A_grow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.77' /' params.py
sed -i -e 's/\(A_melt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.6' /' params.py
python py_mpwp_v0_2.py --out chloe_bc_ag77_am6

sed -i -e 's/\(A_grow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.9' /' params.py
sed -i -e 's/\(A_melt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.7' /' params.py
python py_mpwp_v0_2.py --out chloe_bc_ag9_am7


sed -i -e 's/\(A_grow[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.9' /' params.py
sed -i -e 's/\(A_melt[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '0.2' /' params.py
python py_mpwp_v0_2.py --out chloe_bc_ag9_am2




