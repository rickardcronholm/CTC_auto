#!/usr/bin/env python

# This is the main file for CTC_auto
# CTC_auto is script that converts DICOM files (RP+RD+RS-CT) to a DOSXYZnrc
# compatible egsphant phantom
# It is a modified version of CTC_ask as published in:
# RO Ottosson and CF Behrens. CTC-ask: a new algorithm for conversion of CT
# numbers to tissue parameters for Monte Carlo dose calculations applying
# DICOM RS knowledge. Phys. Med. Biol. 56 (2011) N1-N12
#
#    Copyright [2016] [Rickard Cronholm] Licensed under the
#    Educational Community License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may
#    obtain a copy of the License at
#
#http://www.osedu.org/licenses/ECL-2.0
#
#    Unless required by applicable law or agreed to in writing,
#    software distributed under the License is distributed on an "AS IS"
#    BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#    or implied. See the License for the specific language governing
#    permissions and limitations under the License.
#
# v0.4
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
import os
from scipy import ndimage

# full path to configuration file
confFile = '/home/mcqa/MCQA/samc.conf'

class struct:

    def __init__(self):
        pass

    def init_name(self):
        self.name = []
        self.type = []
        self.logicMatrix = []
        self.materialRamp = []
        self.material = []


def main(RPRDfile, RSfile, CTfile, fileName, addStructType=[], addRampName=[]):
    # set some variables
    with open(confFile) as f:
        confCont = f.readlines()
    whatToGet = ['common.RPfilePrefix', 'common.RDfilePrefix',
        'common.RSfilePrefix', 'common.CTfilePrefix', 'common.DICOMfileEnding',
        'CTC_auto.fixedMedDens', 'CTC_auto.extName',
        'CTC_auto.suppName', 'CTC_auto.suppOuter',
        'CTC_auto.suppInner', 'CTC_auto.densRampName',
        'CTC_auto.externalRamp', 'CTC_auto.outsideRamp',
        'CTC_auto.otherwiseRamp', 'CTC_auto.airDens',
        'CTC_auto.setAir', 'CTC_auto.lowerDens',
        'CTC_auto.spaceDelimit', 'CTC_auto.relElecFile']

    # get the variables from confFile
    cv = struct()
    cv = CTCtools.getConfVars(cv, whatToGet, confCont)

    # define additional variables
    replaceType = 'NONE'  # the structure type to be given to replaced types
    replaceList = []  # list of structure types to be replaced
    extFilterSize = 3  # number of voxel to add to external contour
    extFilt = (extFilterSize, extFilterSize, extFilterSize)
    filt = np.ones(extFilt).astype(int)

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
    # get x and y DICOMcoords in cm
    rd_xmesh, rd_ymesh = CTCtools.getDICOMcoords(rd, True)
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
    if len(np.unique([np.all(np.diff(ct_zmesh) > 0), np.all(np.diff(rd_zmesh) > 0)])) > 1:
        step = np.unique(np.around(np.diff(ct_zmesh), decimals=3))
        if len(step) == 1:
            step = float(step)
        else:
            step = float(np.average(step))
        rd_zmesh = CTCtools.create1dDICOMcoord(rd_zmesh[0], step,
        len(rd_zmesh), -1)
        rd_zmesh = rd_zmesh[::-1]
    '''
    try:
        int(np.argwhere(ct_zmesh == rd_zmesh[-1]))
    except TypeError:
        step = np.unique(np.around(np.diff(ct_zmesh), decimals=3))
        if len(step) == 1:
            step = float(step)
        else:
            step = float(np.average(step))
        rd_zmesh = CTCtools.create1dDICOMcoord(rd_zmesh[0], step,
        len(rd_zmesh), -1)
        rd_zmesh = rd_zmesh[::-1]
    '''

    # inverse arrays and flip matrix depending on meshes
    if ct_xmesh[-1] < ct_xmesh[0]:
        ct_xmesh = ct_xmesh[::-1]
        rd_xmesh = rd_xmesh[::-1]
        ct_mtrx = ct_mtrx[:, :, ::-1]
    if ct_ymesh[-1] < ct_ymesh[0]:
        ct_ymesh = ct_ymesh[::-1]
        rd_ymesh = rd_ymesh[::-1]
        ct_mtrx = ct_mtrx[:, ::-1, :]
    if ct_zmesh[-1] < ct_zmesh[0]:
        ct_zmesh = ct_zmesh[::-1]
        rd_zmesh = rd_zmesh[::-1]
        ct_mtrx = ct_mtrx[::-1, :, :]

    # write doseGrid
    f = open(''.join([fileName, '.doseGrid']), 'w')
    f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_xmesh[0], rd_xmesh[-1],
    len(rd_xmesh)))
    f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_ymesh[0], rd_ymesh[-1],
     len(rd_ymesh)))
    f.write('{0:12.8f} {1:12.8f} {2:d}\n'.format(rd_zmesh[0], rd_zmesh[-1],
    len(rd_zmesh)))
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
    for i in range(0, len(RS.RTROIObservationsSequence)):
        if len(RS.RTROIObservationsSequence[i].RTROIInterpretedType) == 0:
            RS.RTROIObservationsSequence[i].RTROIInterpretedType = replaceType
        try:
            replaceList.index(RS.RTROIObservationsSequence[i].RTROIInterpretedType)
            RS.RTROIObservationsSequence[i].RTROIInterpretedType = replaceType
        except ValueError:
            pass

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

    # Deinterpolate CT data onto dose grid
    # check if rds_zmesh is beyond cts_zmesh, if so eliminate slices
    rds_zmesh = np.intersect1d(np.around(rds_zmesh, decimals=3), np.around(cts_zmesh, decimals=3), assume_unique=True)
    # density matrix is computed using cubic interpolation
    dens_mtrx = CTCtools.map_coordinates(ct_mtrx, cts_xmesh, cts_ymesh,
    cts_zmesh, rds_xmesh, rds_ymesh, rds_zmesh, 3)
    #ct_mtrx = CTCtools.map_coordinates(ct_mtrx, cts_xmesh, cts_ymesh,
    #cts_zmesh, rds_xmesh, rds_ymesh, rds_zmesh, 0)
    # change: use cubic interpolation also for material
    #ct_mtrx = CTCtools.map_coordinates(ct_mtrx, cts_xmesh, cts_ymesh,
    #cts_zmesh, #rds_xmesh, rds_ymesh, rds_zmesh, 3)
    ct_mtrx = copy.deepcopy(dens_mtrx)
    # set nans to 0
    dens_mtrx = np.nan_to_num(dens_mtrx)
    ct_mtrx = np.nan_to_num(ct_mtrx)

    # Locate external contour
    extNr = [i for i, x in enumerate(allTypesFull)
    if x.startswith(cv.extName)][0]
    refROIs.append(int(RS.ROIContourSequence[extNr].ReferencedROINumber))
    refContSeq.append(CTCtools.getCorrContSeq(RS.ROIContourSequence,
    refROIs[-1]))

    # Check if additional structure types were requested
    if len(addStructType) > 0:
        for i in range(0, len(addStructType)):
            addNr = [j for j, x in enumerate(allTypesFull)
            if x == addStructType[i]]
            for j in range(0, len(addNr)):
                refROIs.append(int(RS.ROIContourSequence[addNr[j]].ReferencedROINumber))
                refContSeq.append(CTCtools.getCorrContSeq(RS.ROIContourSequence, refROIs[-1]))

    # Correlate between ROIContourSequence and RTROIObservationsSequence
    refObsSeq = []
    for elem in refROIs:
        refObsSeq.append(CTCtools.getCorrContSeq(RS.RTROIObservationsSequence,
        elem))

    # create and init structures
    structures = []
    structureShell = struct()

    # get name, types and properties
    names = []
    for elem in refObsSeq:
        structure = copy.deepcopy(structureShell)
        # structure.init_name()
        try:
            structure.name = RS.RTROIObservationsSequence[elem].ROIObservationLabel
            names.append(RS.RTROIObservationsSequence[elem].ROIObservationLabel)
        except AttributeError:
            structure.name = ''
            names.append('')
        # names.append(RS.RTROIObservationsSequence[elem].ROIObservationLabel)
        structure.type = RS.RTROIObservationsSequence[elem].RTROIInterpretedType
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
        if structures[cnt].type == cv.extName and not hasattr(structures[cnt],'RelElecDens'):  # add && not releELec !!!
            indx = np.where(ndimage.convolve(structures[cnt].logicMatrix,filt,mode='nearest') >= 1)
            structures[cnt].logicMatrix = np.zeros(structures[cnt].logicMatrix.shape).astype(int)
            structures[cnt].logicMatrix[indx] = 1
        cnt += 1

    # outside
    structures[cnt].logicMatrix = np.ones(structures[0].logicMatrix.shape)

    # make sure that each voxel only belongs to one structure
    # the sorting rules are as follows:
    # If one contour is engulfed by another, the voxels will belong to the engulfed structure
    # If mutual points are found, they are removed from structures in the following order:
    # OUTSIDE, EXTERNAL.
    # If neither of the structures are OUTSIDE or EXTERNAL the voxels are removed from the structure with the lowest index
    if supportStructures:
        # remove suppInner from suppOuter
        inner = [i for i, x in enumerate(names) if cv.suppInner in x][0]
        outer = [i for i, x in enumerate(names) if cv.suppOuter in x][0]
        structures[outer].logicMatrix = np.where(structures[inner].logicMatrix == 1, 0, structures[outer].logicMatrix)
    for i in range(1, len(structures) + 1):
        for j in range(i + 1, len(structures ) + 1):
            if CTCtools.isEngulfed(structures[-j].logicMatrix, structures[-i].logicMatrix):
                structures[-i].logicMatrix = np.where(structures[-j].logicMatrix == 1, 0, structures[-i].logicMatrix)
            else:
                #print '{0:s} is not engulfed by {1:s}'.format(structures[-j].name, structures[-i].name)
                indx = []
                indx = np.where(structures[-i].logicMatrix == structures[-j].logicMatrix)
                numPts = []
                numPts = np.where(structures[-i].logicMatrix[indx] == 1)
                numPts = np.asarray(numPts)
                #print 'Found {0:5d} mutual points'.format(len(np.reshape(numPts,-1)))
                if len(np.reshape(numPts, -1)) > 0:
                    if structures[-j].type == 'OUTSIDE':
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

    # perform HU corrections
    if cv.setAir.lower() == 'y':
        # assumption: the density of air is below the breakpoint in the bilinear HU-dense curve
        airHU = (cv.airDens - densRamp[0][1][0]) / densRamp[0][1][1]
        ct_mtrx = np.where(structures[-1].logicMatrix == 1, airHU, ct_mtrx)
        dens_mtrx = np.where(structures[-1].logicMatrix == 1, airHU, dens_mtrx)

    # compute density matrix
    density = CTCtools.computeDensity(dens_mtrx, densRamp)
    # set minimum density if value is positive
    if cv.lowerDens >= 0:
        density = np.where(density < cv.lowerDens, cv.lowerDens, density)

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
            indx = [j for j, x in enumerate(addStructType)
            if x == structures[i].type][0]
            rampName = addRampName[indx]
        structures[i].ramp = CTCtools.grabData(rampName, 'materialRamp', 0, 2)

    # Build and sort total media list
    medNr, medium = CTCtools.buildGlobalMediaList(structures)

    # compute media matrix
    media = CTCtools.computeMedia(ct_mtrx, structures, medium, medNr)

    # set uniform media and density for structures with defined relElec
    relElec = struct()
    relElec.data = CTCtools.getFromFile(cv.relElecFile, 3)
    relElec.relElec = [x[1] for x in relElec.data]
    relElec.physDens = [x[2] for x in relElec.data]
    relElec.media = [x[0] for x in relElec.data]
    for struc in structures:
        if hasattr(struc, 'RelElecDens'):
            try:
                rIndex = relElec.relElec.index(getattr(struc, 'RelElecDens'))
                # get medNr corresponding to media
                try:
                    medIndx = medium.index(relElec.media[rIndex])
                    media = np.where(struc.logicMatrix == 1, medNr[medIndx],
                    media)
                    density = np.where(struc.logicMatrix == 1,
                    relElec.physDens[rIndex], density)
                except ValueError:  # add if not in list
                    medium.append(relElec.media[rIndex])
                    medNr.append(max(medNr) + 1)
                    media = np.where(struc.logicMatrix == 1, max(medNr),
                    media)
                    density = np.where(struc.logicMatrix == 1,
                    relElec.physDens[rIndex], density)
            except ValueError:
                pass

    # set fixed density for medias listed in cv.fixedMedDens
    if os.path.isfile(cv.fixedMedDens):
        with open(cv.fixedMedDens) as fmd:
            fixedMedDens = fmd.readlines()
        for line in fixedMedDens:
            try:
                fixedMedia = line.split('\t')[0].strip()
                fixedDensity = float(line.split('\t')[1].strip())
                # get medNr corresponding to media
                try:
                    medIndx = medium.index(fixedMedia)
                    density = np.where(media == medNr[medIndx],
                    fixedDensity, density)
                except ValueError:
                    pass
            except ValueError:
                pass
            except IndexError:
                pass

    # rotate phantom and interchange rds_mesh with rds_ymesh if the orientation calls for it
    if orientCT[0] == 0:
        density = density[:, :, ::-1]
        media = media[:, :, ::-1]
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
    if cv.spaceDelimit.lower() == 'y':
        cv.spaceDelimit = True
    else:
        cv.spaceDelimit = False
    print 'Writing egs4phant file'
    CTCtools.writeEgsphant(fileName, rds_xmesh, rds_ymesh, rds_zmesh,
    medium, estepe, media, density, cv.spaceDelimit)


# Function chooser
func_arg = {"-main": main}
# run specifc function as called from command line
if __name__ == "__main__":
    if sys.argv[1] == "-main":
        cnt = 6
        addStructType = []
        addRampName = []
        if len(sys.argv) == 4:
            DICOMdir = sys.argv[2]
            fileName = sys.argv[3]
            # generate RPRD, RS, CT file lists
            with open(confFile) as f:
                confCont = f.readlines()
            whatToGet = ['common.RPfilePrefix', 'common.RDfilePrefix',
            'common.RSfilePrefix', 'common.CTfilePrefix',
            'common.DICOMfileEnding']
            affix = struct()
            affix = CTCtools.getConfVars(affix, whatToGet, confCont)
            CTCtools.genRPRD(DICOMdir, affix.RPfilePrefix, affix.RDfilePrefix,
                affix.DICOMfileEnding, fileName)
            CTCtools.genRS(DICOMdir, affix.RSfilePrefix, affix.DICOMfileEnding,
                fileName)
            CTCtools.genCT(DICOMdir, affix.CTfilePrefix, affix.DICOMfileEnding,
                fileName)
            RPRDfile = os.path.sep.join([DICOMdir, 'RPRD.txt'])
            RSfile = os.path.sep.join([DICOMdir, 'RS.txt'])
            CTfile = os.path.sep.join([DICOMdir, 'CT.txt'])
        else:
            RPRDfile = sys.argv[2]
            RSfile = sys.argv[3]
            CTfile = sys.argv[4]
            fileName = sys.argv[5]
        if len(sys.argv) > cnt:
            while len(sys.argv) > cnt:
                addStructType.append(sys.argv[cnt])
                addRampName.append(sys.argv[cnt + 1])
                cnt += 2
        func_arg[sys.argv[1]](RPRDfile, RSfile, CTfile, fileName, addStructType,
             addRampName)
