#!/bin/bash
# this is an example of looping over a variable and submitting different runs. 
# to change the variable being changed replace 'days' with the variable name and change the values in the for loop 
# to map out a phase space use nested loops.

for i in 100; do
    sed -i -e '/#/!s/\(days[[:space:]]*=[[:space:]]*\)\(.*\)/\1 '$i'. /' params.py
    python py_mpwp_v0.1.py 
    echo "$i"
done
