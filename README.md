# CTC_auto

A python module that converts DICOM CT data to voxalized phantom for the EGSnrc Monte Carlo code DOSXYZnrc

As of version 0.4.0 CTC_auto includes auxilary files needed to run. The auxilary files are site specific, thus the included generic files should only be used as references with regards to structure.

Below is a short description of each auxilary file

ctc_auto.conf
  main configuration files, which is read at run time. The file is read and each requested line is split in variable name and value by white space.
  The structure is as follows:
  Variable value optional comment

All following files are tab (\t) delimited  
cirsLund.dat
  describes the bi-linear relationship between HU and physical density for the CT unit.
  Structure:
  upper bound for fit \t coefficients of linear equation
  
justWater.dat, 24media*.dat, exactIGRT.dat
  describes binning schemes for HU to media assignment.
  Structure:
  media name \t upper bound
  Note that the upper bound must be monotonically increasing
  
SUS.relElec
  Lookup table for assignment of structures that have an assigned relative electron density in the DICOM RS file.
  If the assigned relative electron density is not found in the look up table it will be ignored
  Structure:
  relative electron density \t physical density \t media to assign
  
SUS.fixedMedDens
  Lookup table for assignment of fixed densities to medias
  Structure:
  media name \t physical density

	Copyright [2016] [Rickard Cronholm] Licensed under the
	Educational Community License, Version 2.0 (the "License"); you may
	not use this file except in compliance with the License. You may
	obtain a copy of the License at

http://www.osedu.org/licenses/ECL-2.0

	Unless required by applicable law or agreed to in writing,
	software distributed under the License is distributed on an "AS IS"
	BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
	or implied. See the License for the specific language governing
	permissions and limitations under the License.
