#!/usr/bin/env python

# This is the main file for CTC_auto
# CTC_auto is script that converts DICOM files (RP+RD+RS-CT) to a DOSXYZnrc
# compatible egsphant phantom
# It is a modified version of CTC_ask as published in:
# RO Ottosson and CF Behrens. CTC-ask: a new algorithm for conversion of CT
# numbers to tissue parameters for Monte Carlo dose calculations applying
# DICOM RS knowledge. Phys. Med. Biol. 56 (2011) N1-N12
#
# Rickard Cronholm, 2015
# rickard.cronholm@skane.se
#
# v0.3.7
# Added: 
# Expand CT grid if smaller than Dose Grid
# Sorting rules on voxel belongings (local rules)
# Apply default ramp if no relElecDens specified for structure type NONE
#
# Usage
# python CTC_auto RPRDfile RSfile CTfile fileName
# or
# python CTC_auto DICOMdirectory fileName
#

import sys
import dicom
import numpy as np
import CTCtools
import copy
from scipy import ndimage


class struct:

    def __init__(self):
        pass

    def init_name(self):
        self.name = []
        self.type = []
        self.logicMatrix = []
        self.materialRamp = []
        self.material = []


# set some variables
confFile = '/home/ricr/MCQA/samc.conf'
with open(confFile) as f:
    confCont = f.readlines()
whatToGet = ['common.RPfilePrefix', 'common.RDfilePrefix',
    'common.RSfilePrefix', 'common.CTfilePrefix', 'common.DICOMfileEnding',
    'CTC_askAuto.includeCouch', 'CTC_askAuto.extName', 'CTC_askAuto.suppName',
    'CTC_askAuto.suppOuter', 'CTC_askAuto.suppInner',
    'CTC_askAuto.densRampName', 'CTC_askAuto.externalRamp',
    'CTC_askAuto.outsideRamp', 'CTC_askAuto.otherwiseRamp',
    'CTC_askAuto.airDens', 'CTC_askAuto.setAir', 'CTC_AskAuto.setCouch',
    'CTC_askAuto.densSuppOuter', 'CTC_askAuto.densSuppInner',
    'CTC_askAuto.spaceDelimit', 'CTC_askAuto.relElecFile']

# get the variables from confFile
cv = struct()
cv = CTCtools.getConfVars(cv, whatToGet, confCont)

addStructType = []
addRampName = []

# handle input variables
if len(sys.argv) > 4:  # if usage A
    RPRDfile = sys.argv[1]
    RSfile = sys.argv[2]
    CTfile = sys.argv[3]
    fileName = sys.argv[4]
    cnt = 5
    while len(sys.argv) > cnt:
        addStructType.append(sys.argv[cnt])
        addRampName.append(sys.argv[cnt + 1])
        cnt += 2
else:  # if usage B
    DICOMdirectory = sys.argv[1]
    fileName = sys.argv[2]
    #synthesize RPRDfile, RSfile, CTfile from DICOMdirectory
    CTCtools.genRPRD(DICOMdirectory, cv.RPfilePrefix, cv.RDfilePrefix, cv.DICOMfileEnding, fileName)
    CTCtools.genRS(DICOMdirectory, cv.RSfilePrefix, cv.DICOMfileEnding, fileName)
    CTCtools.genCT(DICOMdirectory, cv.CTfilePrefix, cv.DICOMfileEnding, fileName)
    RPRDfile = ''.join([DICOMdirectory, 'RPRD.txt'])
    RSfile = ''.join([DICOMdirectory, 'RS.txt'])
    CTfile = ''.join([DICOMdirectory, 'CT.txt'])

# load density ramp
# densRamp = CTCtools.getStructFromMatFile(cv.densRampName, 'densRamp', 1)
densRamp = CTCtools.grabData(cv.densRampName, 'densRamp', 1, 1)

# Use DICOM to obtain data

# CT
with open(CTfile) as f:
    ct = f.readlines()
ct = map(str.strip, ct)
# get x and y coords, just use first file in list as all will have same grid
ct_xmesh, ct_ymesh = CTCtools.getDICOMcoords(ct[0], True)  # get DICOMcoords in cm
# get CT matrix and z coords
ct_zmesh, ct_mtrx = CTCtools.getCTinfo(ct, True)

# RD
with open(RPRDfile) as f:
    rp = f.readline().strip()
    rd = f.readline().strip()
# get x and y coords
rd_xmesh, rd_ymesh = CTCtools.getDICOMcoords(rd, True)  # get DICOMcoords in cm
# get z coords
rd_zmesh = CTCtools.getDICOMzCoord(rd, True)

# get orientation of CT and RD
orientCT = CTCtools.getOrient(ct[0])
orientRD = CTCtools.getOrient(rd)

# inverse rd_*mesh depending on orientation relative ct
if sum(orientRD[:3]) == -sum(orientCT[:3]):
    rd_xmesh = rd_xmesh[::-1]
if sum(orientRD[3:]) == -sum(orientCT[3:]):
    rd_ymesh = rd_ymesh[::-1]
try:
    int(np.argwhere(ct_zmesh == rd_zmesh[-1]))
except TypeError:
    step = np.unique(np.around(np.diff(ct_zmesh), decimals=5))
    if len(step) == 1:
        step = float(step)
    else:
        step = float(np.average(step))
    rd_zmesh = CTCtools.create1dDICOMcoord(rd_zmesh[0], step, len(rd_zmesh), -1)
    rd_zmesh = rd_zmesh[::-1]

# inverse arrays and flip matrix depending on meshes
if ct_xmesh[-1] < ct_xmesh[0]:
    ct_xmesh = ct_xmesh[::-1]
    rd_xmesh = rd_xmesh[::-1]
    ct_mtrx = ct_mtrx[:,:,::-1]
if ct_ymesh[-1] < ct_ymesh[0]:
    ct_ymesh = ct_ymesh[::-1]
    rd_ymesh = rd_ymesh[::-1]
    ct_mtrx = ct_mtrx[:,::-1,:]
if ct_zmesh[-1] < ct_zmesh[0]:
    ct_zmesh = ct_zmesh[::-1]
    rd_zmesh = rd_zmesh[::-1]
    ct_mtrx = ct_mtrx[::-1,:,:]

# write doseGrid
f = open(''.join([fileName, '.doseGrid']), 'w')
f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_xmesh[0], rd_xmesh[-1], len(rd_xmesh)))
f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_ymesh[0], rd_ymesh[-1], len(rd_ymesh)))
f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_zmesh[0], rd_zmesh[-1], len(rd_zmesh)))
f.close()

cts_xmesh = ct_xmesh[:]
cts_ymesh = ct_ymesh[:]
cts_zmesh = ct_zmesh[:]
rds_xmesh = rd_xmesh[:]
rds_ymesh = rd_ymesh[:]
rds_zmesh = rd_zmesh[:]


refROIs = []
refContSeq = []

# Read RS file
with open(RSfile) as f:
    rs = f.readline().strip()
RS = dicom.read_file(rs)  # open file
# replace empty strucutre types with specified string
replaceType = 'NONE'
for i in range(0, len(RS.RTROIObservationsSequence)):
    if len(RS.RTROIObservationsSequence[i].RTROIInterpretedType) == 0:
        RS.RTROIObservationsSequence[i].RTROIInterpretedType = replaceType

# start reading RS struct types
allTypesFull = []
for i in range(0, len(RS.RTROIObservationsSequence)):
    allTypesFull.append(RS.RTROIObservationsSequence[i].RTROIInterpretedType)
allTypes = list(set(allTypesFull))
# Check for support structures
try:
    allTypes.index(cv.suppName)
    supportStructures = True
except ValueError:
    supportStructures = False

if supportStructures:
    suppDim = struct()
    suppDim.X = [None] * 2
    suppDim.Y = [None] * 2
    suppDim.Z = [None] * 2
    # get indices of support structures
    suppNr = [i for i, x in enumerate(allTypesFull) if x == cv.suppName]
    # get the ref ROI number(s)
    for elem in suppNr:
        refROIs.append(int(RS.ROIContourSequence[elem].ReferencedROINumber))
    # find the corresponding ContourSequence
    for elem in refROIs:
        refContSeq.append(CTCtools.getCorrContSeq(RS.ROIContourSequence, elem))
    # get the extreme of the support structures
    for elem in refContSeq:
        suppDim = CTCtools.getExtremeOfContour(CTCtools.getContour(RS.ROIContourSequence[elem], ct_zmesh, abs(orientCT[1]), True), suppDim)
    # extend rd_*mesh to encompass support structures
    rds_xmesh = CTCtools.extendMesh(rd_xmesh, suppDim.X)
    rds_ymesh = CTCtools.extendMesh(rd_ymesh, suppDim.Y)
    rds_zmesh = CTCtools.extendMesh(rd_zmesh, suppDim.Z)
    suppDim.X = [rds_xmesh[0], rds_xmesh[-1]]
    suppDim.Y = [rds_ymesh[0], rds_ymesh[-1]]
    suppDim.Z = [rds_zmesh[0], rds_zmesh[-1]]
    cts_xmesh = CTCtools.extendMesh(ct_xmesh, suppDim.X)
    cts_ymesh = CTCtools.extendMesh(ct_ymesh, suppDim.Y)
    cts_zmesh = CTCtools.extendMesh(ct_zmesh, suppDim.Z)
    # pad ct using map_coordinates if cts != ct
    if ct_mtrx.shape != (len(cts_zmesh), len(cts_ymesh), len(cts_xmesh)):
        ct_mtrx = CTCtools.map_coordinates(ct_mtrx, ct_xmesh, ct_ymesh, ct_zmesh, cts_xmesh, cts_ymesh, cts_zmesh, 0)

# extend CT if rds_*mesh is beyond cts_*mesh
ct0_xmesh = cts_xmesh[:]
ct0_ymesh = cts_ymesh[:]
ct0_zmesh = cts_zmesh[:]
if np.min(rds_xmesh) < np.min(cts_xmesh) or np.max(rds_xmesh) > np.max(cts_xmesh):
    limits = [np.min([np.min(rds_xmesh), np.min(cts_xmesh)]), np.max([np.max(rds_xmesh), np.max(cts_xmesh)])]
    cts_xmesh = CTCtools.extendMesh(cts_xmesh, limits)
if np.min(rds_ymesh) < np.min(cts_ymesh) or np.max(rds_ymesh) > np.max(cts_ymesh):
    limits = [np.min([np.min(rds_ymesh), np.min(cts_ymesh)]), np.max([np.max(rds_ymesh), np.max(cts_ymesh)])]
    cts_ymesh = CTCtools.extendMesh(cts_ymesh, limits)
if np.min(rds_zmesh) < np.min(cts_zmesh) or np.max(rds_zmesh) > np.max(cts_zmesh):
    limits = [np.min([np.min(rds_zmesh), np.min(cts_zmesh)]), np.max([np.max(rds_zmesh), np.max(cts_zmesh)])]
    cts_zmesh = CTCtools.extendMesh(cts_zmesh, limits)
# pad ct using map_coordinates if cts != ct
if ct_mtrx.shape != (len(cts_zmesh), len(cts_ymesh), len(cts_xmesh)):
    ct_mtrx = CTCtools.map_coordinates(ct_mtrx, ct0_xmesh, ct0_ymesh, ct0_zmesh, cts_xmesh, cts_ymesh, cts_zmesh, 0)


# Deinterpolate CT data onto dose grid
# check if rds_zmesh is beyond cts_zmesh, if so eliminate slices
rds_zmesh = np.intersect1d(np.around(rds_zmesh, decimals = 5), np.around(cts_zmesh, decimals = 5), assume_unique = True)
# density matrix is computed using cubic interpolation
dens_mtrx = CTCtools.map_coordinates(ct_mtrx, cts_xmesh, cts_ymesh, cts_zmesh,
rds_xmesh, rds_ymesh, rds_zmesh, 3)
ct_mtrx = CTCtools.map_coordinates(ct_mtrx, cts_xmesh, cts_ymesh, cts_zmesh,
rds_xmesh, rds_ymesh, rds_zmesh, 0)
# set nans to 0
dens_mtrx = np.nan_to_num(dens_mtrx)
ct_mtrx = np.nan_to_num(ct_mtrx)

# Locate external contour
extNr = [i for i, x in enumerate(allTypesFull) if x.startswith(cv.extName)][0]
refROIs.append(int(RS.ROIContourSequence[extNr].ReferencedROINumber))
refContSeq.append(CTCtools.getCorrContSeq(RS.ROIContourSequence, refROIs[-1]))

# Check if additional structure types were requested
if len(addStructType) > 0:
    for i in range(0, len(addStructType)):
        addNr = [j for j, x in enumerate(allTypesFull) if x == addStructType[i]]
        for j in range(0, len(addNr)):
            refROIs.append(int(RS.ROIContourSequence[addNr[j]].ReferencedROINumber))
            refContSeq.append(CTCtools.getCorrContSeq(RS.ROIContourSequence, refROIs[-1]))

# Correlate structures between ROIContourSequence and RTROIObservationsSequence
refObsSeq = []
for elem in refROIs:
    refObsSeq.append(CTCtools.getCorrContSeq(RS.RTROIObservationsSequence, elem))

# create and init structures
structures = []
structureShell = struct()

# get name, types and properties
names = []
types = []
for elem in refObsSeq:
    structure = copy.deepcopy(structureShell)
    # structure.init_name()
    structure.name = RS.RTROIObservationsSequence[elem].ROIObservationLabel
    names.append(RS.RTROIObservationsSequence[elem].ROIObservationLabel)
    structure.type = RS.RTROIObservationsSequence[elem].RTROIInterpretedType
    types.append(RS.RTROIObservationsSequence[elem].RTROIInterpretedType)
    try:
        if RS.RTROIObservationsSequence[elem].ROIPhysicalPropertiesSequence[0].ROIPhysicalProperty == 'REL_ELEC_DENSITY':
            structure.RelElecDens = float(RS.RTROIObservationsSequence[elem].ROIPhysicalPropertiesSequence[0].ROIPhysicalPropertyValue)
    except AttributeError:
        pass
    #if not structure.type == cv.suppName:
    #structure.RelElecDens = 1.0000  # set to 1 for all structures
    structures.append(structure)

# create and append structure for outside
structure = copy.deepcopy(structureShell)
names.append('OUTSIDE')
types.append('OUTSIDE')
structure.name = 'OUTSIDE'
structure.type = 'OUTSIDE'
structures.append(structure)

# get contour sequences and create logicMatrix
cnt = 0

for elem in refContSeq:
    # get contour
    structures[cnt].contour = CTCtools.getContour(RS.ROIContourSequence[elem], ct_zmesh, abs(orientCT[1]), True)
    # deInterpolate contour onto dose grid and generate boolean matrix
    structures[cnt].logicMatrix = CTCtools.interpStructToDose(structures[cnt].contour, rds_xmesh, rds_ymesh, rds_zmesh, cts_xmesh, cts_ymesh, cts_zmesh)
    # expand external contour by convolution
    if structures[cnt].type == cv.extName:  # add && not releELec !!!
        #print type(structures[cnt].logicMatrix)
        filt = np.ones((3,3,3)).astype(int)
        #numPts = np.where(structures[cnt].logicMatrix == 1)
        #numPts = np.asarray(numPts)
        #print 'Before convolution {0:d} points'.format(len(np.reshape(numPts,-1)))
        #structures[cnt].logicMatrix = ndimage.convolve(structures[cnt].logicMatrix,filt,mode='nearest')
        indx = np.where(ndimage.convolve(structures[cnt].logicMatrix,filt,mode='nearest') >= 1)
        structures[cnt].logicMatrix = np.zeros(structures[cnt].logicMatrix.shape).astype(int)
        structures[cnt].logicMatrix[indx] = 1
        #numPts = []
        #numPts = np.where(structures[cnt].logicMatrix == 1)
        #numPts = np.asarray(numPts)
        #print 'After convolution {0:d} points'.format(len(np.reshape(numPts,-1)))
        #print 'Total points {0:d}'.format(structures[cnt].logicMatrix.size)
    cnt += 1

# outside
structures[cnt].logicMatrix = np.ones(structures[0].logicMatrix.shape)

# make sure that each voxel only belongs to one structure
if supportStructures:
    # remove suppInner from suppOuter
    inner = [i for i, x in enumerate(names) if x.startswith(cv.suppInner)][0]
    outer = [i for i, x in enumerate(names) if x.startswith(cv.suppOuter)][0]
    structures[outer].logicMatrix = np.where(structures[inner].logicMatrix == 1, 0, structures[outer].logicMatrix)

# remove external from outside
extNr = [i for i, x in enumerate(types) if x == cv.extName][0]
outNr = [i for i, x in enumerate(types) if x == 'OUTSIDE'][0]
structures[outNr].logicMatrix = np.where(structures[extNr].logicMatrix == 1, 0, structures[outNr].logicMatrix)

# reverse order of structures in order to handle EXTERNAL prior to OUTSIDE
structures = structures[::-1]

for i in range(1, len(structures) + 1):
    for j in range(i+1, len(structures ) + 1):
        if CTCtools.isEngulfed(structures[-j].logicMatrix, structures[-i].logicMatrix):
            #print '{0:s}.{2:s} is engulfed by {1:s}.{3:s}'.format(structures[-j].name, structures[-i].name, structures[-j].type, structures[-i].type)
            if structures[-j].type == 'CAVITY' and structures[-i].type == 'OUTSIDE':
                structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == 1, 0, structures[-j].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-j].name)
            elif structures[-i].type == 'CAVITY' and structures[-j].type == 'OUTSIDE':
                structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == 1, 0, structures[-i].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-i].name)
            else:
                structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == 1, 0, structures[-i].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-i].name)
        elif CTCtools.isEngulfed(structures[-i].logicMatrix, structures[-j].logicMatrix):
            #print '{0:s}.{2:s} is engulfed by {1:s}.{3:s}'.format(structures[-i].name, structures[-j].name, structures[-i].type, structures[-j].type)
            if structures[-i].type == 'CAVITY' and structures[-j].type == 'OUTSIDE':
                structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == 1, 0, structures[-i].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-i].name)
            elif structures[-j].type == 'CAVITY' and structures[-i].type == 'OUTSIDE':
                structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == 1, 0, structures[-j].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-j].name)
            else:
                structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == 1, 0, structures[-j].logicMatrix)
                #print 'removing mutual points from {0:s}'.format(structures[-j].name)
        else:
            #print '{0:s}.{2:s} is not engulfed by {1:s}.{3:s} and vice versa'.format(structures[-j].name, structures[-i].name, structures[-j].type, structures[-i].type)
            indx = []
            indx = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix)
            numPts = []
            numPts = np.where(structures[-i].logicMatrix[indx] == 1)
            numPts = np.asarray(numPts)
            #print 'Found {0:5d} mutual points'.format(len(np.reshape(numPts,-1)))
            if len(np.reshape(numPts,-1)) > 0:
                if structures[-j].type == 'CAVITY':
                    structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix, 0, structures[-j].logicMatrix)
                    #print 'removing mutual points from {0:s}'.format(structures[-j].name)
                elif structures[-i].type == 'CAVITY':
                    structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == structures[-i].logicMatrix, 0, structures[-i].logicMatrix)
                    #print 'removing mutual points from {0:s}'.format(structures[-i].name)
                elif structures[-j].type == 'OUTSIDE':
                    structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix, 0, structures[-j].logicMatrix)
                    #print 'removing mutual points from {0:s}'.format(structures[-j].name)
                elif structures[-i].type == 'OUTSIDE':
                    structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == structures[-i].logicMatrix, 0, structures[-i].logicMatrix)
                    #print 'removing mutual points from {0:s}'.format(structures[-i].name)
                else:
                    if structures[-j].type == cv.extName:
                        structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix, 0, structures[-j].logicMatrix)
                    elif structures[-i].type == cv.extName:
                        structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == structures[-i].logicMatrix, 0, structures[-i].logicMatrix)
                    else:
                        structures[-j].logicMatrix = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix, 0, structures[-j].logicMatrix)
                    #print 'removing mutual points from {0:s}'.format(structures[-j].name)

# get relElec corrections
relElec = CTCtools.getFromFile(cv.relElecFile, 0)

# perform HU corrections
if cv.setAir.lower() == 'y':
    # assumption: the density of air is below the breakpoint in the bilinear HU-dense curve
    airHU = (cv.airDens - densRamp[0][1][0])/densRamp[0][1][1]
    ct_mtrx = np.where(structures[-1].logicMatrix == 1, airHU, ct_mtrx)
    dens_mtrx = np.where(structures[-1].logicMatrix == 1, airHU, dens_mtrx)
for i in range(0, len(structures)):
    try:
        k = structures[i].RelElecDens
        # perform RelElecDens corrections
        for j in range(0, len(relElec)):
            if np.around(structures[i].RelElecDens, decimals=5) == np.around(relElec[j][0], decimals=5):
                myDens = relElec[j][1]
                myHU = CTCtools.getHUfromDens(myDens, densRamp)
                ct_mtrx = np.where(structures[i].logicMatrix == 1, myHU, ct_mtrx)
                dens_mtrx = np.where(structures[i].logicMatrix == 1, myHU, dens_mtrx)
                break
    except AttributeError:
        pass

# compute density matrix
density = CTCtools.computeDensity(dens_mtrx, densRamp)

# assign material ramps to each structure
for i in range(0, len(structures)):
    if structures[i].type == cv.extName:
        rampName = cv.externalRamp
    elif structures[i].type == cv.suppName:
        rampName = cv.otherwiseRamp
    elif structures[i].type == 'OUTSIDE':
        rampName = cv.outsideRamp
    else:
        # find correct rampName
        indx = [j for j, x in enumerate(addStructType) if x == structures[i].type][0]
        rampName = addRampName[indx]
        if structures[i].type == 'NONE':
            try:
                k = structures[i].RelElecDens
            except AttributeError:
                rampName = cv.externalRamp
    #structures[i].ramp = CTCtools.getStructFromMatFile(rampName, 'materialRamp', 0)
    structures[i].ramp = CTCtools.grabData(rampName, 'materialRamp', 0, 2)

# convert matlab type material ramp to pythonic arrays
#for i in range(0, len(structures)):
#    structures[i].ramp = CTCtools.convMatToPyth(structures[i].ramp)

# Build and sort total media list
medNr, medium = CTCtools.buildGlobalMediaList(structures)

# compute media matrix
media = CTCtools.computeMedia(ct_mtrx, structures, medium, medNr)

# rotate phantom and interchange rds_mesh with rds_ymesh if the orientation calls for it
if orientCT[0] == 0:
    density = density[:,:,::-1]
    media = media[:,:,::-1]
    tmp = rds_ymesh[:]
    rds_ymesh = rds_xmesh[:]
    rds_xmesh = tmp[:]
    density = np.transpose(density, (2, 1, 0))
    density = np.rot90(density, 3)
    density = np.transpose(density, (2, 1, 0))
    media = np.transpose(media, (2, 1, 0))
    media = np.rot90(media, 3)
    media = np.transpose(media, (2, 1, 0))

# write to file
estepe = [0.25] * len(medium)  # dummy variable for ESTEPE
# f = open(''.join([fileName, '.doseGrid']), 'w')
# f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_xmeshO[0], rd_xmeshO[-1], len(rd_xmeshO)))
# f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_ymeshO[0], rd_ymeshO[-1], len(rd_ymeshO)))
# f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_zmeshO[0], rd_zmeshO[-1], len(rd_zmeshO)))
# f.close()
if cv.spaceDelimit.lower() == 'y':
    cv.spaceDelimit = True
else:
    cv.spaceDelimit = False
print 'Writing egs4phant file'
CTCtools.writeEgsphant(fileName, rds_xmesh, rds_ymesh, rds_zmesh, medium, estepe, media, density, cv.spaceDelimit)
