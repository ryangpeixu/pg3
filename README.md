# PG3: Policy-Guided Planning for Generalized Policy Generation
This repository implements the main approach described in the IJCAI 2022 paper: 

PG3: Policy-Guided Planning for Generalized Policy Generation. 

Ryan Yang*, Tom Silver*, Aidan Curtis, Tomas Lozano-Perez, Leslie Kaelbling

For any questions or issues with the code, please email ryanyang@mit.edu and tslvr@mit.edu.

Link to paper: *Coming soon*

Instructions for running (tested on OS X and Ubuntu 18.04):
* Use Python 3.6 or higher.
* Download Python dependencies: `pip install -r requirements.txt`.

Now, `./run.sh` should work. Different environments can be run by
changing the ENV variable in run.sh. The expected result on the
default setting (spannerlearning) is that performance on the test problems
for "Integrated canonical match" should be 0.0 at the very start, and get
to 1.0 by training iteration 100. Test performance can be found by looking
for the printout "Evaluation results:"; testing is run every 100 iterations.
