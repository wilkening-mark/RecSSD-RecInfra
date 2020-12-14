import os, sys

#for model in ['dien',
#              'din',
#              'mtwnd',
#              'ncf',
#              'rm1',
#              'rm2',
#              'rm3',
#              'wnd']:
for model in ['rm3', 'wnd']:
    command = "python sweep_" + model + ".py > sweep_" + model + "_base_out.txt"
    print("-------------------- Running " + model + " --------------------\n")
    os.system(command)
