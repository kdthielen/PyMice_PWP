Code using Python 2.7.15

I've separated this into 3 files for what I think is ease of use/readability:

1) py_mpwp_v0.py
2) params.py
3) pypwp_functions.py


These need to be in the same directory (or alter the import statements).

Forcing and initial conditions are included (also same directory or need to point to where you've moved them)

To run simply $python py_mpwp_v0.py 

Let me know for any desired changes/things that are hard to understand - I expect these so don't hold back!


Short file description below----------------------------------------------------


1) the main file where pwp is run, makes subdriectories and computes stuff. default is to make a subdirectory where it is being run of the form pypwp_*date of simulation* with variables used in that run and a further subdirectory to save sim output. 

2) Params.py - paramater file, separated into switches (Bulk/gradient richardson, Krauss Turner, Louise ice, and some antiquated internal wave thing that I've commented out in (1) (uconn)

3) the functions being used in the 1d model, and to write initial variable metadata. this probably shouldn't be altered unless there is a bug or if you are adding features. Really only seperate to make (1) more readable. 





