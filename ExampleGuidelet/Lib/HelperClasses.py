import logging
import os
import time
import numpy as np
import re
import slicer, vtk

class Session(object):
    def __init__(self,userDataDict, listOfRecordings=[]):
        logging.debug('Session object init()')
        self.userDataDict = userDataDict
        self.sessionCreationTime = time.strftime(r"%Y-%m-%d-%H%M%S")
        self.sessionSavedFlag = False
        self.savedFilePathName = None # empty until saved
        self.listOfRecordings = listOfRecordings

    def saveToFile(self, saveDir):
        logging.debug('Session.saveToFile()')
        saveFileName = self.getSaveFileName()
        saveFileText = self.getSaveFileText()
        saveFilePathName = os.path.join(saveDir, saveFileName)
        with open(saveFilePathName, 'w') as f:
            f.write(saveFileText)    
        self.sessionSavedFlag = True # once saved to file 
        self.savedFilePathName = saveFilePathName

    def getSaveFileText(self):
        logging.debug('Session.getSaveFileText()')
        headerTextList = [val for val in self.userDataDict.values()]
        delimLine = "---" # delimiter line separating header from the rest
        recordingsLines = [recObj.recordingFilePath for recObj in self.listOfRecordings]
        saveFileText = "\n".join([*headerTextList,delimLine,*recordingsLines]) +"\n"
        return saveFileText

    def getSaveFileName(self):
        logging.debug('Session.getSaveFileName()')
        prefix = 'Session'
        userName = self.userDataDict['userName']
        saveFileName = f"{prefix}-{userName.replace(' ','_')}-{self.sessionCreationTime}.txt"
        return saveFileName
    
    def addRecording(self, newRecordingObject, updateSavedFileIfAlreadySaved=False):
        logging.debug('Session.addRecording')
        self.listOfRecordings.append(newRecordingObject)
        if updateSavedFileIfAlreadySaved and self.savedFilePathName:
            saveDir = os.path.dirname(self.savedFilePathName)
            self.saveToFile(saveDir)
    
    def removeRecording(self, recordingObjectToRemove, updateSavedFileIfAlreadySaved=False):
        self.listOfRecordings.remove(recordingObjectToRemove)
        if updateSavedFileIfAlreadySaved and self.savedFilePathName:
            saveDir = os.path.dirname(self.savedFilePathName)
            self.saveToFile(saveDir)

    def getListOfRecordingFileNames(self):
        listOfRecordingFileNames = [R.recordingFilePath for R in self.listOfRecordings]
        return listOfRecordingFileNames
    
    def getListOfScopeRuns(self):
        logging.debug('Session.getListOfScopeRuns()')
        listOfScopeRunsForSession = []
        for R in self.listOfRecordings:
            for S in R.listOfScopeRuns:
                listOfScopeRunsForSession.append(S)
        logging.debug(f'  {len(listOfScopeRunsForSession)} runs found in total')
        return listOfScopeRunsForSession



class Recording(object):
    def __init__(self, parentSessionObject, recordingFilePath, listOfScopeRuns=[]):
        logging.debug('Recording object init()')
        self.parentSession = parentSessionObject
        self.recordingFilePath = recordingFilePath # should include full path and file name
        self.listOfScopeRuns = listOfScopeRuns

    def processRecordingToScopeRuns(self, sceneLeafTransformNode, segmentationNode, airwayZoneSegmentName='airwayZone'):
        logging.debug('Recording.processRecordingToScopeRuns()')
        scopeRuns = self.processRecordingFileToScopeRuns(self.recordingFilePath, sceneLeafTransformNode, segmentationNode, airwayZoneSegmentName)
        for scopeRun in scopeRuns:
            scopeRun.setParentRecordingObject(self)

    @classmethod 
    def processRecordingFileToScopeRuns(cls, recordingFilePath, sceneLeafTransformNode, segmentationNode, airwayZoneSegmentName='airwayZone'):
        logging.debug('Recording.processRecordingFileToScopeRuns()')
        logging.info(f"  Processing {recordingFilePath}...\n   Using {sceneLeafTransformNode.GetName()} as transform leaf\n   Using {segmentationNode.GetName()} as segmentation\n")
        timeStamps, headSensorTransforms, scopeSensorTransforms = cls.import_tracker_recording(recordingFilePath)
        transformsList, transformNames = gatherTransformsFromTransformHierarchy(leafTransformNode=sceneLeafTransformNode)
        # TODO: Verify that transformNames[1] looks like it's the dynamic head sensor and [2] looks like the dynamic scope sensor
        # Replace the single transform matrix arrays in transformsList with the full set from 
        # the loaded file for both the head sensor and scope sensor
        HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 1
        SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 2
        # Replace the single transform at each appropriate location with the full sequence of 
        # sensor transforms loaded from the recording file
        transformsList[HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = headSensorTransforms
        transformsList[SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = scopeSensorTransforms
        # Find sequence of positions/orientations using the t
        positions, orientations = positions_from_transform_hierarchy(transformsList)
        # Break this sequence into separate runs
        runsData = identifyTrackingRunsFromRawPath(positions, segmentationNode, airwayZoneSegmentName)
        # runsData is a list, where each element is a list of indices associtated with a single run
        # indices are for timeStamps, positions, and orientations
        scopeRuns = []
        for runData in runsData:
            runData = np.array(runData)
            runPositions = positions[runData, : ]
            runOrientations = orientations[runData, : ]
            runTimeStamps = timeStamps[runData, : ]
            scopeRun = ScopeRun(None, runTimeStamps, runPositions, runOrientations)
            scopeRuns.append(scopeRun)
        return scopeRuns

    
    

    @classmethod
    def import_tracker_recording(cls, mha_file_path):
      """Import the sequence of transforms stored in one of the guidelet mhd files.
      """     
      # Can be used as Recording.import_tracker_recording()
      logging.debug(f'Recording.import_tracker_recording({mha_file_path})')
      logging.debug(f'  Opening file: {mha_file_path} ...')
      # Just parse far enough to get number of timesteps
      numSeqFrames = 0
      with open(mha_file_path) as f:
          for line in f:
              if line.startswith('DimSize'):
                  numSeqFrames = int(line.rstrip().split()[-1])
                  logging.debug(f'  Found DimSize line: {numSeqFrames} time steps in file')
                  break
      if numSeqFrames==0:
          raise(Exception('DimSize line not found in mha file processing, cannot determine number of processed time steps.'))
      # Preallocate transform arrays
      timeStamps = np.zeros((numSeqFrames))
      headSensorTransforms = np.zeros((4,4,numSeqFrames))
      scopeSensorTransforms = np.zeros((4,4,numSeqFrames))
      # Sequence line info pattern
      transformLinePatt = re.compile(r'Seq_Frame(?P<frameNumber>\d\d\d\d)_(?P<transformName>\w+) =(?P<matrix>( -?\d+(\.\d+)?([Ee][+-]?\d+)?){16})')
      timeStampLinePatt = re.compile(r'Seq_Frame(?P<frameNumber>\d\d\d\d)_Timestamp = (?P<timeStamp>-?\d+(\.\d+)?)')
      # Read the file line by line
      with open(mha_file_path) as f:
          keepgoing = True
          for line in f:
              if m := transformLinePatt.search(line):
                  # Process the result
                  groupDict = m.groupdict()
                  #logging.debug(groupDict.__str__())
                  frameNum = int(groupDict['frameNumber'])
                  transformName = groupDict['transformName']
                  transformMatrixList = groupDict['matrix'].lstrip().split() # row order 
                  transformMatrix = np.array(transformMatrixList).reshape((4,4))
                  # Store in arrays
                  if re.search('Head', transformName):
                      headSensorTransforms[:,:,frameNum] = np.linalg.inv(transformMatrix.astype('float64'))  # THESE NEED TO BE INVERTED!!!
                  elif re.search('Stylus', transformName):
                      scopeSensorTransforms[:,:,frameNum] = transformMatrix
              if m := timeStampLinePatt.search(line):
                  groupDict = m.groupdict()
                  frameNum = int(groupDict['frameNumber'])
                  timeStamp = float(groupDict['timeStamp'])
                  timeStamps[frameNum] = timeStamp
      return timeStamps, headSensorTransforms, scopeSensorTransforms
    


class ScopeRun(object):
    def __init__(self, parentRecordingObject, timestamps, positions, orientations):
        logging.debug('ScopeRun object init()')
        self.parentRecording = parentRecordingObject
        self.timeStamps = timestamps
        self.positions = positions
        self.orientations = orientations

    def setParentRecordingObject(self, parentRecordingObject):
        logging.debug('ScopeRun.setParentRecordingObject()')
        self.parentRecording = parentRecordingObject

    def saveToFile(self, saveDir):
        logging.debug('ScopeRun.saveToFile()')


## Helper functions not tied to a class or instance

def gatherTransformsFromTransformHierarchy(leafTransformNode):
    """ Use the existing scene transform hierarchy to build a list of 
    transformation matrices (called transformsList)
    above a given leaf transform node (i.e. the list returned will be all
    transforms above the leaf (inclusive))
    """
    logging.debug('HelperClasses.gatherTransformsFromTransformHierarchy()')
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    leafTransformNode.GetTransformNodeID()
    curT = leafTransformNode
    transformNodeList = [leafTransformNode]
    while curT.GetTransformNodeID():
      # Get parent transform node
      curT = slicer.mrmlScene.GetNodeByID(curT.GetTransformNodeID())
      transformNodeList.append(curT)
    # Reverse order so decending hierarchy rather than ascending
    transformNodeList.reverse()
    transformNames = [tNode.GetName() for tNode in transformNodeList]
    transformList = [slicer.util.arrayFromTransformMatrix(tNode) for tNode in transformNodeList]
    return  transformList, transformNames

def positions_from_transform_hierarchy(transformsList):
    """Compute series of locations given a list of hierarchical transform matrices. 
    transformsList[i] must either be a single 4x4 array or a 4x4xN array, where N is 
    the number of time step frames. transformsList[i] is the parent transform to 
    transformsList[i+1]
    """
    numFramesPer = np.zeros((len(transformsList)),dtype=int)
    for idx, tListItem in enumerate(transformsList):
        numDims = tListItem.ndim
        if numDims==3:
            numFramesPer[idx] = tListItem.shape[2]
        elif numDims==2:
            numFramesPer[idx] = 1
    # NumFramesPer elements should now all either equal 1 or the same number of frames
    numFrames = np.max(numFramesPer)
    assert np.all((numFramesPer==1) | (numFramesPer==numFrames) ), 'All transforms supplied must have either a single frame or the same number of frames!'        
    # Calculate positions from the sequence of transforms
    positions = np.zeros((numFrames, 3))
    orientations = np.zeros((numFrames,3))
    origin = np.zeros((4))
    origin[3] = 1 # homogenous coordinate for a point
    direction = np.zeros((4))
    direction[2] = 1 # [0,0,1,0]
    direction[3] = 0 # homogenous coord for a vector
    concatTransforms = []
    for frameNum in range(numFrames):
        # Assemble the correct list of transforms for this frame
        currentTransformList = []
        for listIdx in range(len(transformsList)):
            if numFramesPer[listIdx]==1:
                curTransform = transformsList[listIdx]
            else:
                curTransform = transformsList[listIdx][:, :, frameNum]
            currentTransformList.append(curTransform)
        # Apply tranforms in order to origin position
        concatTransform = np.linalg.multi_dot(currentTransformList)
        concatTransforms.append(concatTransform)
        currentPosition4 = concatTransform @ origin
        currentOrientation4 = concatTransform @ direction
        positions[frameNum,:] = currentPosition4[0:3]
        orientations[frameNum,:] = currentOrientation4[0:3]
    return positions, orientations

def identifyTrackingRunsFromRawPath(positions, segmentationNode, airwayZoneSegmentName, minimumRunLengthPoints=100):
    """Using a segmentation node which has a segment for the airway zone, combined
    with a set of sequentual positions, divide these positions in to "runs".  
    A run is a (mostly) contiguous section of the position indices which includes at least one
    airwayZone point and all surrounding airwayZone points. 
    Runs are merged if they have less than a 10 point gap between them (assumed due to 
    aberrent sensor readings)
    """
    logging.debug('HelperClasses.identifyTrackingRunsFromRawPath()')
    runsData = []
    #points = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
    points = positions
    nPoints = points.shape[0]
    segmentNames = getSegmentNamesAtRasPoint(segmentationNode, points)
    # Make a mask of airwayZone points
    airwayZoneMask = np.zeros((nPoints))    
    for idx in range(nPoints):
        segNamesNow = segmentNames[idx]
        airwayZoneMask[idx] = 1 if airwayZoneSegmentName in segNamesNow else 0

    # If no airwayZone points, no runs
    if np.all(airwayZoneMask==0):
        logging.info(f'No runs present because no points are inside {airwayZoneSegmentName} segment!')
        return []
    
    while np.any(airwayZoneMask==1):
        # Find first deep point index
        firstDeepIdx = np.argmax(airwayZoneMask)
        startIdx = firstDeepIdx
        # Run forwards to find the last contiguous point which is still in zone
        stopIdx = startIdx+1
        while True:
            if stopIdx==len(airwayZoneMask) or (airwayZoneMask[stopIdx] != 1):
                break
            else: 
                # increment
                stopIdx = stopIdx + 1
        # Store data
        runData = list(range(startIdx, stopIdx))
        logging.debug(f'Run identified from index {startIdx} to {stopIdx}.')
        runsData.append(runData)
        # Clear this run from deepMask
        airwayZoneMask[startIdx:stopIdx] = 0
    # Merge any runs which are separated by only a short break (assume the gap is 
    # due to an aberrent sensor reading)
    RUN_GAP_TOLERANCE = 10 # at normal sampling rates, this represents about 1/3 second
    runDataGaps = np.array([runsData[idx+1][0] - runsData[idx][-1] for idx in range(len(runsData)-1)])
    while np.any(runDataGaps <= RUN_GAP_TOLERANCE):
        # Merge the first one and then see if any remain
        firstGapBelowTolIdx = np.nonzero(runDataGaps <= RUN_GAP_TOLERANCE)[0]
        runsData[firstGapBelowTolIdx] = [*runsData[firstGapBelowTolIdx], *runsData[firstDeepIdx+1]]
        del runsData[firstGapBelowTolIdx+1] # remove duplicate of merged section
        logging.debug(f"  Merged run {firstGapBelowTolIdx} and {firstGapBelowTolIdx+1} because gap was < {RUN_GAP_TOLERANCE} points!")
        runDataGaps = np.array([runsData[idx+1][0] - runsData[idx][-1] for idx in range(len(runsData)-1)])
    # Enforce minimum run lengths
    runsData = [runData for runData in runsData if len(runData)>=minimumRunLengthPoints]
    
    return runsData

def getSegmentNamesAtRasPoint(segmentationNode,rasPoints=[[0,0,0],[1,1,1]], includeHiddenSegments=True, sliceViewLabel='Green'):
    """ Returns names of segments at the rasPoint location.  If includeHiddenSegments is false (default is True)
    then only currently visible segments (in the first display node) will be included as possible outputs. 
    If true (now the default), then all segments will be included, regardless of current visibility. It is possible 
    to specify which slice you want to do the query in by specifying a different sliceViewLabel.  I have not
    tested whether the slice view needs to be included in the current layout or if it needs to be visible.
    rasPoints can be a list of 3 element lists or an nx3 numpy array.
    
    Important NOTE: GetVisibleSegmentsForPosition() only identifies segments which are VISIBLE and are IN THE SLICE PLANE of 
    the selected slice view label.  It is not good enough that the segmentation visibility is turned
    on there, the segment itself must show up in that slice plane (though it doesn't seem to need to
    actually be showing on the screen; for example, if you zoom in or pan that segment off the side of 
    the slice view it still works, but if you scroll away to a slice plane which does not contain the 
    segment it does not work). This function gets around that limitation by 1) jumping the slice view
    to the plane containing the rasPoint and 2) creating a temporary segmentation display node which 
    ensures that the segmentation is visible in the selected slice view.
    """
    ##sliceNode = slicer.mrmlScene.GetNodeByID(f'vtkMRMLSliceNode{sliceViewLabel}')
    sliceViewWidget = slicer.app.layoutManager().sliceWidget(sliceViewLabel)
    # Store the old offset so that we can reset to this after jumping
    sliceNode = sliceViewWidget.mrmlSliceNode()
    oldOffset = sliceNode.GetSliceOffset()
    # Ensure segmentation is visible in this slice widget (otherwise the list will never return any segment names)
    tempDisplayNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationDisplayNode')
    if not includeHiddenSegments:
        # Copy everything (crucially, including current segment visibility settings)
        tempDisplayNode.Copy(segmentationNode.GetDisplayNode()) 
        # If this is not done, all segments are visible by default, and therefore will be included in the output
    tempDisplayNode.SetVisibility3D(0) # no need to show in 3D
    tempDisplayNode.SetVisibility2D(1)
    tempDisplayNode.SetViewNodeIDs((sliceNode.GetID(),))
    tempDisplayNode.SetVisibility(1)
    segmentationNode.AddAndObserveDisplayNodeID(tempDisplayNode.GetID())
    
    segmentationsDisplayableManager = sliceViewWidget.sliceView().displayableManagerByClassName("vtkMRMLSegmentationsDisplayableManager2D")
    # Loop over ras points
    segmentNames = []
    nPoints = len(rasPoints)
    for pointIdx in range(nPoints):
        rasPoint = rasPoints[pointIdx]
        # Jump to slice containing query point
        sliceNode.JumpSliceByOffsetting(*rasPoint)
        # Get list of segment names at that point
        segmentIds = vtk.vtkStringArray()
        segmentationsDisplayableManager.GetVisibleSegmentsForPosition(rasPoint, tempDisplayNode, segmentIds)
        segmentNamesForCurrentPoint = [segmentationNode.GetSegmentation().GetSegment(segmentIds.GetValue(idx)).GetName()  for idx in range(segmentIds.GetNumberOfValues())]
        segmentNames.append(segmentNamesForCurrentPoint)
    # Restore prior state
    sliceNode.SetSliceOffset(oldOffset)
    segmentationNode.RemoveNthDisplayNodeID(segmentationNode.GetNumberOfDisplayNodes()-1)
    return segmentNames