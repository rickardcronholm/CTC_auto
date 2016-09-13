# Toolbox for CTC_auto
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

import glob
import scipy.io
import dicom
import numpy as np
import copy
from progressbar import ProgressBar
from scipy import ndimage, ogrid, mgrid, interpolate
from skimage import draw
from scipy.interpolate import InterpolatedUnivariateSpline


class struct:

    def __init__(self):
        pass


def getConfVars(cv, wtg, content):
    for item in wtg:  # loop over what to get
        for elem in content:  # loop over content
            if elem.startswith(item):  # if found
                name = elem.split()[0].split('.')[-1]
                val = elem.split()[1]
                try:
                    val = float(val)
                except ValueError:
                    pass
                setattr(cv, name, val)  # add to confvariables
    return cv


def genRPRD(dicomDir, RPFilePrefix, RDFilePrefix, DICOMFileEnding, fileName):
    content = []
    # get full paths
    rp = getFromGlob(dicomDir, RPFilePrefix, DICOMFileEnding)
    rd = getFromGlob(dicomDir, RDFilePrefix, DICOMFileEnding)
    # create list
    content.append(rp[0].strip())
    content.append(rd[0].strip())
    # write file
    writeContFile(dicomDir, 'RPRD.txt', content)


def genRS(dicomDir, RSFilePrefix, DICOMFileEnding, fileName):
    content = []
    # get full paths
    rs = getFromGlob(dicomDir, RSFilePrefix, DICOMFileEnding)
    # create list
    content.append(rs[0].strip())
    # write file
    writeContFile(dicomDir, 'RS.txt', content)


def genCT(dicomDir, CTFilePrefix, DICOMFileEnding, fileName):
    content = []
    # get full paths
    ct = getFromGlob(dicomDir, CTFilePrefix, DICOMFileEnding)
    # create list
    for item in ct:
        content.append(item.strip())
    # write file
    writeContFile(dicomDir, 'CT.txt', content)


def getFromGlob(dicomDir, prefix, suffix):
    return glob.glob(''.join([dicomDir, prefix, '*', suffix]))


def writeContFile(dicomDir, fileName, content):
    f = open(''.join([dicomDir, fileName]), 'w')
    for element in content:
        f.write('{0}\n'.format(element))


def getStructFromMatFile(fileName, attr, level):
    data = struct()
    matfile = scipy.io.loadmat(fileName, squeeze_me=True,
    struct_as_record=False)
    if level == 0:
        for field in matfile[attr]._fieldnames:
            val = getattr(matfile[attr], field)
            setattr(data, field, val)

    if level == 1:
        for elem in matfile[attr]:
            for field in elem._fieldnames:
                try:
                    cont = []
                    val = getattr(data, field)
                    cont.append(val)
                    val = getattr(elem, field)
                    cont.append(val)
                    setattr(data, field, cont)
                except AttributeError:
                    val = getattr(elem, field)
                    setattr(data, field, val)

    return data


def getDICOMcoords(fileName, cm):
    dcm = dicom.read_file(fileName)  # read DICOM file

    # get directional cosines
    xDir = map(int, dcm.ImageOrientationPatient[:3])
    yDir = map(int, dcm.ImageOrientationPatient[3:])

    # get xCoord
    if abs(xDir[0]) == 1:
        xCoord = getCoord(sum(xDir), float(dcm.ImagePositionPatient[0]),
        float(dcm.PixelSpacing[0]), dcm.Columns)
    else:
        xCoord = getCoord(sum(xDir), float(dcm.ImagePositionPatient[1]),
        float(dcm.PixelSpacing[1]), dcm.Columns)
    xCoord = np.asarray(xCoord)

    # get yCoord
    if abs(yDir[1]) == 1:
        yCoord = getCoord(sum(yDir), float(dcm.ImagePositionPatient[1]),
        float(dcm.PixelSpacing[1]), dcm.Rows)
    else:
        yCoord = getCoord(sum(yDir), float(dcm.ImagePositionPatient[0]),
        float(dcm.PixelSpacing[0]), dcm.Rows)
    yCoord = np.asarray(yCoord)

    # if cm is 1 convert to cm from DICOM native mm
    if cm:
        xCoord[:] = [x / 10 for x in xCoord]
        yCoord[:] = [x / 10 for x in yCoord]

    return xCoord, yCoord


def getCoord(dircos, position, spacing, elements):
    coord = np.linspace(position, position + (elements-1) * dircos * spacing, num = elements)
    return coord


def getDICOMzCoord(fileName, cm):
    dcm = dicom.read_file(fileName)  # read DICOM file
    if dcm.Modality == 'CT':
        zCoord = dcm.ImagePositionPatient[2]
    else:
        zCoord = np.asarray(dcm.GridFrameOffsetVector)
        zCoord[:] = [x + dcm.ImagePositionPatient[2] for x in zCoord]

    # if cm is 1 convert to cm from DICOM native mm
    if cm:
        zCoord[:] = [x / 10 for x in zCoord]

    return zCoord


def getCTinfo(fileList, cm):
    # define static variables
    ctOffset = 1000
    zCoord = np.zeros(len(fileList))

    # loop through fileList to get individual z coord
    for i in range(0, len(fileList)):
        zCoord[i] = getDICOMzCoord(fileList[i], False)  # get z coord in mm
    if cm:
        zCoord[:] = [x / 10 for x in zCoord]  # convert to cm

    # sort by ascending
    ct_order = np.argsort(zCoord)
    # arrange zCoord
    zCoordArr = np.zeros(len(zCoord))
    for i in range(0, len(ct_order)):
        zCoordArr[i] = zCoord[ct_order[i]]

    # get CT data
    for i in range(0, len(fileList)):
        ct_slice = getCTdata(fileList[ct_order[i]])
        if i == 0:

            ct_mtrx = np.empty((len(zCoord), len(ct_slice), len(ct_slice[0])))
        ct_mtrx[:][:][i] = ct_slice

    # apply ctOffset
    ct_mtrx[:] = [x + ctOffset for x in ct_mtrx]

    return zCoordArr, ct_mtrx


def getCTdata(fileName):
    dcm = dicom.read_file(fileName)  # read DICOM file
    ct_slice = np.ndarray.astype(dcm.pixel_array, 'int')
    ct_slice[:] = [x * int(dcm.RescaleSlope) + int(dcm.RescaleIntercept)
    for x in ct_slice]
    return ct_slice


def getOrient(fileName):
    dcm = dicom.read_file(fileName)  # open file
    return map(int, dcm.ImageOrientationPatient)


def create1dDICOMcoord(start, pix, dim, direction):
    stop = start + direction * pix * (dim - 1)
    return np.linspace(start, stop, dim)


def deInterpCTmatrix(ct_mtrx, ct_xmesh, ct_ymesh, ct_zmesh, rd_xmesh, rd_ymesh, rd_zmesh, method):
    pbar = ProgressBar()
    mtrx = np.empty((len(rd_zmesh), len(rd_ymesh), len(rd_xmesh)))
    grid_x, grid_y = np.mgrid[rd_xmesh[0]:rd_xmesh[-1]:len(rd_xmesh) * 1j,
    rd_ymesh[0]:rd_ymesh[-1]:len(rd_ymesh) * 1j]

    points = np.empty((len(ct_xmesh) * len(ct_ymesh), 2))
    cnt = 0
    for elem in ct_xmesh:
        for item in ct_ymesh:
            points[cnt][0] = elem
            points[cnt][1] = item
            cnt += 1

    for i in pbar(range (0, len(rd_zmesh))):
        sliceNr = np.argwhere(np.around(ct_zmesh, decimals=5) ==
        np.around(rd_zmesh[i], decimals=5))[0][0]
        ct_slice = np.reshape(ct_mtrx[sliceNr][:][:].T, -1)

        mtrx[i][:][:] = interpolate.griddata(points, ct_slice,
        (grid_x, grid_y), method).T

    return mtrx


def getContour(RSData, zmesh, flip, cm):
# create and initialize structures
    contour = [None] * len(zmesh)

    for i in range(0, len(RSData.ContourSequence)):
        rawCont = map(float, RSData.ContourSequence[i].ContourData)
        if cm:
            rawCont[:] = [x / 10 for x in rawCont]  # convert to cm
        xCont = np.zeros(len(rawCont) / 3)
        yCont = np.zeros(len(rawCont) / 3)
        zCont = np.zeros(len(rawCont) / 3)
        for j in range(1, len(xCont) + 1):
            xCont[j - 1] = rawCont[3 * j - 3]
            yCont[j - 1] = rawCont[3 * j - 2]
            zCont[j - 1] = rawCont[3 * j - 1]
        # add the first element to form closed loop
        xCont = np.append(xCont, xCont[0])
        yCont = np.append(yCont, yCont[0])
        zCont = np.append(zCont, zCont[0])

        if flip == 0:
            cont = np.vstack((xCont, yCont, zCont))
        else:
            cont = np.vstack((yCont, xCont, zCont))

        # locate which slice nr it belongs to
        sliceNr = np.argwhere(np.around(zmesh, decimals=5) ==
        np.around(cont[2][0], decimals=5))[0][0]

        # the first data set will implicitly belong to a new slice
        if i == 0 or sliceNr != lastSlice:
            segments = []
        points = cont[:]
        segments.append(points)
        contour[sliceNr] = segments[:]

        lastSlice = sliceNr

    return contour


def getExtremeOfContour(data, extremes):
    # stack data
    x = np.empty(0)
    y = np.empty(0)
    z = np.empty(0)
    for i in range(0, len(data)):
        if data[i] is not None:
            for j in range(0, len(data[i])):
                x = np.hstack((x, data[i][j][0]))
                y = np.hstack((y, data[i][j][1]))
                z = np.hstack((z, data[i][j][2]))

    # compare extremes
    extremes.X = compExtremes(extremes.X, x)
    extremes.Y = compExtremes(extremes.Y, y)
    extremes.Z = compExtremes(extremes.Z, z)

    return extremes


def compExtremes(ext, arr):
    if ext[0] is None:
        ext[0] = min(arr)
    else:
        ext[0] = min(ext[0], min(arr))
    if ext[1] is None:
        ext[1] = max(arr)
    else:
        ext[1] = max(ext[1], max(arr))
    return ext


def getCorrContSeq(seq, lookfor):
    search = 0
    while True:
        if int(seq[search].ReferencedROINumber) == lookfor:
            break
        search += 1

    return search


def extendMesh(mesh, ext):
    meshMin = min(mesh)
    meshMax = max(mesh)
    step = np.unique(np.around(np.diff(mesh), decimals=5))[0]
    low = 0
    high = 0
    if ext[0] < meshMin:
        while meshMin > ext[0]:
            meshMin -= step
            low += 1
    if ext[1] > meshMax:
        while meshMax < ext[1]:
            meshMax += step
            high += 1

    if high > 0:
        highMesh = np.arange(1, high + 1)
        highMesh = highMesh * step + mesh[-1]
        mesh = np.hstack([mesh, highMesh])
    if low > 0:
        lowMesh = np.arange(low, 0, -1)
        lowMesh = lowMesh * -step + mesh[0]
        mesh = np.hstack([lowMesh, mesh])

    return mesh


def interpStructToDose(contour, rd_x, rd_y, rd_z, ct_x, ct_y, ct_z):
    # start by generating a mask based on CT grid
    maskM = np.zeros((len(ct_z), len(ct_y), len(ct_x)))
    xL = np.arange(len(ct_x))
    yL = np.arange(len(ct_y))

    # iterate through contour to generate a mask for each slice
    # note; there may be empty slices as well as slices with multiple segments
    pbar = ProgressBar()
    for j in pbar(range(0, len(contour))):
        if contour[j] is not None:  # skip empty slices
            sliceNr = np.argwhere(np.around(ct_z, decimals=5) ==
            np.around(contour[j][0][2][0], decimals=5))[0][0]
            for i in range(0, len(contour[j])):
                # take care of unbound contours
                points_x = copy.deepcopy(contour[j][i][0][:])  # get x-points
                points_y = copy.deepcopy(contour[j][i][1][:])  # get y-points
                points_x[np.where(points_x < min(ct_x))] = min(ct_x)
                points_x[np.where(points_x > max(ct_x))] = max(ct_x)
                points_y[np.where(points_y < min(ct_y))] = min(ct_y)
                points_y[np.where(points_y > max(ct_y))] = max(ct_y)
                # interpolate to voxel number instead of absolute coord
                points_x = np.interp(points_x, ct_x, xL)
                points_y = np.interp(points_y, ct_y, yL)
                # assign mask based on polygons
                [rr, cc] = draw.polygon(np.asarray(points_y),
                np.asarray(points_x), (len(ct_y), len(ct_x)))
                tempMask = np.zeros((len(ct_y), len(ct_x)))
                tempMask[rr, cc] = 1
                # add for the current slice
                maskM[sliceNr][:][:] = np.add(maskM[sliceNr][:][:],
                tempMask[:][:])

    del tempMask
    maskM = np.where(maskM > 1, 1, maskM)  # remove duplicates

    # interpolate maskM on to the RD grid
    order = 1
    mask = map_coordinates(maskM, ct_x, ct_y, ct_z, rd_x, rd_y, rd_z, order)
    return np.around(mask).astype(int)


def cropCT(mtrx, ct_x, ct_y, ct_z, rd_x, rd_y, rd_z):
    margin = 5
    # x
    ind = getIndx(ct_x, rd_x, margin)
    ct_x = ct_x[ind[0]:ind[1] + 1]
    mtrx = mtrx[:, :, ind[0]:ind[1] + 1]
    # y
    ind = getIndx(ct_y, rd_y, margin)
    ct_y = ct_y[ind[0]:ind[1] + 1]
    mtrx = mtrx[:, ind[0]:ind[1] + 1, :]
    # z
    ind = getIndx(ct_z, rd_z, margin)
    ct_z = ct_z[ind[0]:ind[1] + 1]
    mtrx = mtrx[ind[0]:ind[1] + 1, :, :]

    return mtrx, ct_x, ct_y, ct_z


def getIndx(a, b, margin):
    ind = []
    if a[0] < a[-1]:
        ind.append((np.abs(a - b[0])).argmin() - margin)
        ind.append((np.abs(a - b[-1])).argmin() + margin)
    else:
        ind.append((np.abs(a - b[-1])).argmin() + margin)
        ind.append((np.abs(a - b[0])).argmin() - margin)

    ind = np.asarray(ind)
    ind = ind.clip(min=0)
    ind = ind.clip(max=len(a) - 1)

    return ind


def map_coordinates(mtrx, ct_x, ct_y, ct_z, rd_x, rd_y, rd_z, order):
    mtrxOut = np.zeros((len(rd_z), len(rd_y), len(rd_x)))  # preallocate

    # create sparse grids
    xct, yct = ogrid[min(ct_x):max(ct_x):len(ct_x) * 1j,
    min(ct_y):max(ct_y):len(ct_y) * 1j]

    # interpolate coord arrays in cm to voxel nr
    xL = np.linspace(0, len(ct_x) - 1, len(ct_x))
    yL = np.linspace(0, len(ct_y) - 1, len(ct_y))
    extrapolatorX = InterpolatedUnivariateSpline(ct_x, xL, k=1)
    x = extrapolatorX(rd_x)
    extrapolatorY = InterpolatedUnivariateSpline(ct_y, yL, k=1)
    y = extrapolatorY(rd_y)
    # create mesh grids
    xrd, yrd = mgrid[min(x):max(x):len(x) * 1j, min(y):max(y):len(y) * 1j]
    coords = np.array([yrd, xrd])

    pbar = ProgressBar()
    # map coordinates slice by slice
    cnt = 0
    for i in pbar(range (0, len(rd_z))):
        try:
            sliceNr = np.argwhere(np.around(ct_z,
            decimals=5) == np.around(rd_z[i], decimals=5))[0][0]
            ct_slice = ndimage.map_coordinates(mtrx[sliceNr][:][:],
            coords, order=order).T
        except IndexError:
            ct_slice = np.zeros((len(rd_y), len(rd_x)))
        mtrxOut[cnt][:][:] = ct_slice[:][:]
        cnt += 1

    return mtrxOut


def computeDensity(mtrx, ramp):
    # remove nans
    mtrx = np.nan_to_num(mtrx)
    densMtrx = np.zeros(mtrx.shape)
    # compute density for all but last 'segment'
    for i in range(0, len(ramp) - 1):
        densMtrx = np.where(mtrx <= ramp[i][0],
        mtrx * ramp[i][1][1] + ramp[i][1][0], densMtrx)
    # compute density for last 'segment'
    densMtrx = np.where(mtrx > ramp[-2][0], mtrx * ramp[-1][1][1] +
    ramp[-1][1][0], densMtrx)
    # remove negatives
    densMtrx = np.where(densMtrx < 0., 0., densMtrx)
    return densMtrx


def computeMedia(mtrx, structures, medium, medNr):
    # remove nans
    mtrx = np.nan_to_num(mtrx)
    medMtrx = np.zeros(mtrx.shape)
    # compute media for each structure in a loop
    for elem in structures:
        tempMtrx = np.zeros(mtrx.shape)
        names = []
        bounds = []
        # loop through appending names and bounds
        for i in range(0, len(elem.ramp)):
            names.append(elem.ramp[i][0])
            bounds.append(elem.ramp[i][1])
        # sort according to bounds
        indx = [bounds.index(x) for x in sorted(bounds)]
        names[:] = [names[x] for x in indx]
        bounds[:] = [bounds[x] for x in indx]
        # find global medNr
        globalNr = medNr[medium.index(names[0])]
        tempMtrx = np.where(mtrx <= bounds[0], globalNr, tempMtrx)
        for i in range(1, len(elem.ramp)):
            globalNr = medNr[medium.index(names[i])]
            tempMtrx = np.where(mtrx > bounds[i - 1], globalNr, tempMtrx)

        medMtrx = np.where(elem.logicMatrix == 1, tempMtrx, medMtrx)
    return medMtrx.astype('int')


def convMatToPyth(indata):
    outData = []
    materials = []
    mat_ct_up_bound = []
    if type(indata.materials) == type(unicode('A')):
        materials.append(indata.materials)
        mat_ct_up_bound.append(indata.mat_ct_up_bound)
    else:
        for i in range(0, len(indata.materials)):
            materials.append(indata.materials[i])
            mat_ct_up_bound.append(indata.mat_ct_up_bound[i])
    materials = map(str, materials)
    outData.append(map(str.strip, materials))
    outData.append(mat_ct_up_bound)
    return outData


def buildGlobalMediaList(structures):
    # loop over structures to get all names
    medNames = []
    for i in range(0, len(structures)):
        for j in range(0, len(structures[i].ramp)):
            medNames.append(structures[i].ramp[j][0].strip())
    medNames = flatten(medNames)
    medNames = list(set(medNames))

    # sort medium based on average upper ct bound
    summation = [0] * len(medNames)
    average = [0] * len(medNames)
    for i in range(0, len(medNames)):
        cnt = 0
        for j in range(0, len(structures)):
            for k in range(0, len(structures[j].ramp)):
                try:
                    indx = structures[j].ramp[k][0].index(medNames[i])
                    summation[i] += structures[j].ramp[k][1]
                    cnt += 1
                except ValueError:
                    pass
        average[i] = float(summation[i]) / cnt

    sortIndx = np.ndarray.tolist(np.argsort(np.asarray(average)))
    medium = [medNames[x] for x in sortIndx]

    # generate medNr list
    medNr = np.linspace(1, len(medium), len(medium)).astype('int')
    medNr = np.ndarray.tolist(medNr)

    return medNr, medium


def writeEgsphant(fileName, x, y, z, medium, estepe, media, density, spaceDelimit):
    halfThick = False
    xb = createBoundGrid(x)
    yb = createBoundGrid(y)
    zb = createBoundGrid(z)

    if halfThick:
        # "interpolate" to half slice thickness
        zn = np.linspace(z[0], z[-1], 2 * len(z) - 1)
        z = copy.deepcopy(zn)
        zb = createBoundGrid(z)

    f = open(''.join([fileName, '.egs4phant']), 'w')
    # Write number of media and their names
    f.write('{0:d}\n'.format(len(medium)))
    for elem in medium:
        f.write('{}\n'.format(elem))
    for elem in estepe:
        f.write('{:6.2f}'.format(elem))  # write estepe
    f.write('\n')
    # write num elements and their boundaries
    f.write('{0:5d}{1:5d}{2:5d}'.format(len(x), len(y), len(z)))
    # write x bound
    writeBound(f, xb, 6)
    # write y bound
    writeBound(f, yb, 6)
    # write z bound
    writeBound(f, zb, 6)
    f.write('\n')
    # write media matrix
    pbar = ProgressBar()
    if spaceDelimit:
        for i in pbar(range(0, len(z))):
            for j in range(0, len(y)):
                for k in range(0, len(x)):
                    if not halfThick:
                        # for normal thick
                        f.write('{0:d} '.format(media[i][j][k]))
                    else:
                        # for half thick
                        f.write('{0:d} '.format(media[int(i / 2)][j][k]))
                f.write('\n')
            f.write('\n')
    else:
        for i in pbar(range(0, len(z))):
            for j in range(0, len(y)):
                for k in range(0, len(x)):
                    if not halfThick:
                        # for normal thickness
                        f.write('{0:d}'.format(media[i][j][k]))
                    else:
                        # for half thick
                        f.write('{0:d}'.format(media[int(i / 2)][j][k]))
                f.write('\n')
            f.write('\n')
    # write density matrix
    for i in range(0, len(z)):
        for j in range(0, len(y)):
            if not halfThick:
                writeBound(f, density[i][j][:], 5)  # for normal thickness
            else:
                writeBound(f, density[int(i / 2)][j][:], 5)  # for half thick
        f.write('\n')

    f.close()


def writeBound(f, grid, num):
    for i in range(0, len(grid)):
        if np.mod(i, num) == 0:
            f.write('\n')
        f.write('{0:12.8f} '.format(grid[i]))


def createBoundGrid(grid):
    step = np.unique(np.around(np.diff(grid), decimals=5))[0]
    return np.linspace(grid[0] - step / 2, grid[-1] + step / 2, len(grid) + 1)


def getFromFile(fileName, switch):
    myList = []
    with open(fileName) as f:
        data = f.readlines()

    data = map(str.strip, data)

    if switch == 0:
        for elem in data:
            try:
                myList.append(map(float, elem.split()))
            except ValueError:
                pass
    elif switch == 1:
        il = []
        for elem in data:
            try:
                map(float, elem.split())
                innerList = copy.deepcopy(il)
                innerList.append(float(elem.split()[0]))
                innerList.append(np.asarray(map(float, elem.split()[1:])))
                myList.append(innerList)
            except ValueError:
                pass
    elif switch == 2:
        il = []
        for elem in data:
            try:
                int(elem.split()[1])
                innerList = copy.deepcopy(il)
                innerList.append(elem.split()[0])
                innerList.append(int(elem.split()[1]))
                myList.append(innerList)
            except ValueError:
                pass
    elif switch == 3:
        for elem in data:
            try:
                elems = filter(None, elem.split('\t'))
                myList.append([elems[2].strip(), float(elems[0].strip()),
                float(elems[1].strip())])
            except ValueError:
                pass
            except IndexError:
                pass

    return myList


def grabData(fileName, attr, level, switch):
    data = getFromFile(fileName, switch)
    if len(data) == 0:
        data = getStructFromMatFile(fileName, attr, level)
        data = getPythonicList(data, attr)

    return data


def getPythonicList(data, attr):
    myList = []
    tempList = []
    il = []
    if attr == 'materialRamp':
        lookFor = ['materials', 'mat_ct_up_bound']

    else:
        lookFor = ['upperCTbound', 'coefficients']

    for i in range(0, len(lookFor)):
        tempList.append(getattr(data, lookFor[i]))

    # reaarrange
    for i in range(0, len(tempList[0])):
        innerList = copy.deepcopy(il)
        try:
            innerList.append(float(tempList[0][i]))
        except ValueError:
            innerList.append(str(tempList[0][i]).strip())
        innerList.append(tempList[1][i])
        myList.append(innerList)

    return myList


def flatten(seq, container=None):
    if container is None:
        container = []
    for s in seq:
        if hasattr(s, '__iter__'):
            flatten(s, container)
        else:
            container.append(s)
    return container


def getHUfromDens(dens, densRamp):
    # iteratively compare against boundaries in densRamp
    cnt = 0
    while cnt < len(densRamp):
        myHU = (dens - densRamp[cnt][1][0]) / densRamp[cnt][1][1]
        if myHU <= densRamp[cnt][0]:
            break
        cnt += 1

    return myHU


def isEngulfed(A, B):
    A = np.reshape(A, -1)
    B = np.reshape(B, -1)
    # return True if all points in A exists in B
    indx = np.where(A == 1)
    check = np.unique(B[indx] == 1)
    if len(check) > 1 or len(check) == 0:
        check = False
    else:
        check = check[0]

    return check
