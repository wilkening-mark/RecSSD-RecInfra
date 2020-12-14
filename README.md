RecSSD-RecInfra: Benchmark End-to-end DNN-based Recommendation Systems
======================================================================

RecSSD-RecInfra presents a modified version of DeepRecInfra, integrating SSD NDP
techniques to explore the performance of recommendation inference using
advanced SSD systems.

Make sure that ./models/libflashrec.so is copied correctly from the
RecSSD-UNVMeDriver repository, and the OpenSSD device is properly bound to
the driver.

Running Sweeps
==============

Before running sweeps, set up the input traces by using the bash script
./models/input/create_dist.sh. Changing the value of K defined at the top of
./models/input/gen_dist.py will adjust the locality distribution used to generate
the synthetic input trace data.

Within the ./models/ directory, use the sweep_{benchmark}_.py scripts to run
benchmark models across a range of batch sizes. Individual model parameters can
be edited within the scripts by adjusting the command line parameters passed to
dlrm_s_caffe2.py. Adjusting sls_type="base" or sls_type="ndp" will change whether
or not the model uses standard baseline SSD interfaces or our NDP interfaces to
implement the SparseLengthSum operator and the reading of categorical features
from the embedding tables.
