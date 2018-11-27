Some small changes to previous version.

-- in py_mpwp file

1) Cleaned up some commenting 
2) Added option for command line input of the following 
  - output directory name (--out) defaults to pypwp_DDMMYY_HHMM
  - filename of profiles (--fname) defaults to profiles_#.npz
  - paramater filename (--params) defaults to params.py 
  - initial profile data (--init) defaults to AS_LCDW.mat
  - met forcing input (--force) defaults to AS_longmet.mat
  
3) Added sea ice salinity in KT/BC ice bits if BC turned off then A=0 so will not play a roll in KT

4) Copies the used paramater file to output directory instead of rewriting selection of inputs. 

-- in param file 
1) slightly organized to include params affected by switches below switches. Moved some constants to functions file
such as g, cp, Latent heat values.

-- in functions file
1) moved some constants here as changed import scheme of main to allow arbitrary paramater file name.
Probably could do this better? 

ADDED FILES

Test.mat is matlab output for comparison

sed_examp.sh is example of bash commands to use to automate changes to param file

compare_pwp examp of accessing/plotting the matlab and python output. 

