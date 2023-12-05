Steps to retrieve the intrinsic parameters of the scanner:
  1) Connect to the scanner in PhoXi Control
  2) Trigger the scan
  3) Open terminal
  4) change the directory to the one where intrinsic_parameters.sh is located
  5) run the command:
	bash intrinsic_parameters.sh
The intrinsic parameters will be saved into the file intrinsic_parameters.txt
The camera matrix values are written in succession by rows from left to right.
The distortion coefficients are in the order k1,k2,p1,p2,k3. The rest are zeros.
All values in length are in millimeters.
All of the coefficients are compatible with the OpenCV standard.

In case you get an error regarding the libtiff library then run the command 
	sudo apt-get install libtiff5
this will install the neccesary library, then run command 5) again.
More information: support@photoneo.com
