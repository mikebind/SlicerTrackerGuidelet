import logging
import os
import time
import numpy as np
import json
import re
import slicer, vtk
from typing import List, Tuple, Optional, Union
import scipy
from pathlib import Path


"""
General Overview
---------------- 
Session objects are associated with each time the Save User Info button 
is clicked.  Each recording is processed and added to the current session 
upon stopping.  The guidelet saves the session to disk after every recording
completion. 
While in memory, a Session object keeps track of a list of Recording objects.
When loaded from a file, these objects are not reconstituted, and the 
listOfRecordings property is just empty (I think). 

When a recording is stopped, the Recording object processes the raw data
into individual scope runs, using recording file and the current transform
hierarchy to generate raw path data, and then the 
identifyTrackingRunsFromRawPath function to divide the raw path into segments
which represent individual scope runs.  The basic criteria used is that a scope
run is composed of a position which is inside the airwayZone segment plus all 
points neighboring time points which are also inside the airwayZone segment or 
which leave the airwayZone segment for fewer than 11 consecutive timepoints.  
Also, if the identified run length is less then the minimum of 30 consecutive
time points, it is also discarded. 

Therefore, the processing to scope runs should do a good job of weeding out
stretches of time where the scope tip is not actually in or very near the airway.
Also, it should not be thrown off by an isolated errant point location.  Beyond 
that, it should leave all location data intact.  Specifically, it doesn't do 
anything like progress fraction filtering, that's all further along in the processing
stream. Scope runs are intended to be pretty raw data. 
"""


class Session(object):
    def __init__(self, userDataDict, listOfRecordings=None):
        logging.debug("Session object init()")
        self.userDataDict = userDataDict
        self.sessionCreationTime = time.strftime(r"%Y-%m-%d-%H%M%S")
        self.sessionSavedFlag = False
        self.savedFilePathName = None  # empty until saved
        self.listOfRecordings = listOfRecordings or []  # default to empty list

    @classmethod
    def loadFromFile(cls, filePathName):
        """Code looks non-functional..."""
        logging.debug(f'Session.loadFromFile("{filePathName}")')
        with open(filePathName, "r") as f:
            # Parse file text
            fileContent = f.readlines()
        delimPatt = re.compile("---")
        recordingSectionFirstLinePatt = re.compile(".*")
        currentSection = "header"
        for line in fileContent:
            if delimPatt.match(line):
                # Changing sections
                currentSection = "changing"
            if currentSection == "header":
                # set up userDataDict
                userDataDict = {"userName": line.strip()}
            elif currentSection == "changing":
                # inspect first line of new section
                if delimPatt.match(line):
                    # shouldn't happen, but just say we're still changing
                    pass
                elif recordingSectionFirstLinePatt.match(line):
                    currentSection = "recording"
                elif scopeRunSectionFirstLinePatt.match(line):
                    currentSection = "scopeRun"
                else:
                    raise Exception(
                        f'First line of section "{line.strip()}" does not match any expected section patterns!'
                    )

        # parse file name for creation time (this should maybe included in the file text)
        filename = os.path.basename(filePathName)
        return sess

    def saveToFile(self, saveDir=None):
        logging.debug("Session.saveToFile()")
        if saveDir is None and self.sessionSavedFlag:
            saveDir = os.path.dirname(self.savedFilePathName)
        saveFileName = self.getSaveFileName()
        saveFileText = self.getSaveFileText()
        saveFilePathName = os.path.join(saveDir, saveFileName)
        with open(saveFilePathName, "w") as f:
            f.write(saveFileText)
        self.sessionSavedFlag = True  # once saved to file
        self.savedFilePathName = saveFilePathName

    def getSaveFileText(self):
        """Save file text has a header section, then zero or more
        recording sections, each of which can contain zero or more
        scope run sections.
        Recording sections have a recording file path line,
        a list of transform hierarchy names line,
        and then a list of transform matrices
        Scope run sections have a header line ("JSON formatted list of run data. [timeStamp, pos_R,..."),
        and then a line with all of the time, position, oriZ and oriX data
        """
        logging.debug("Session.getSaveFileText()")
        headerTextList = [
            self.userDataDict["userName"]
        ]  # NOTE: used to be [val for val in self.userDataDict.values()]
        delimLine = "---"  # delimiter line separating header from the rest
        sections = []
        sections.extend(headerTextList)
        sections.append(delimLine)
        for recObj in self.listOfRecordings:
            section = []
            # Record file path and then scope run data
            recObjFilePath = recObj.recordingFilePath
            section.append(recObjFilePath)
            section.append(recObj.getTransformsInfoString())
            section.append(delimLine)
            for scopeRun in recObj.listOfScopeRuns:
                scopeRunText = scopeRun.getSaveDataText()
                section.append(scopeRunText)
                section.append(delimLine)
            sections.extend(section)
        saveFileText = "\n".join(sections)
        return saveFileText

    def getSaveFileName(self):
        logging.debug("Session.getSaveFileName()")
        prefix = "Session"
        userName = self.userDataDict["userName"]
        saveFileName = (
            f"{prefix}-{userName.replace(' ','_')}-{self.sessionCreationTime}.txt"
        )
        return saveFileName

    def addRecording(self, newRecordingObject, updateSavedFileIfAlreadySaved=False):
        logging.debug("Session.addRecording")
        self.listOfRecordings.append(newRecordingObject)
        if updateSavedFileIfAlreadySaved and self.savedFilePathName:
            saveDir = os.path.dirname(self.savedFilePathName)
            self.saveToFile(saveDir)

    def removeRecording(
        self, recordingObjectToRemove, updateSavedFileIfAlreadySaved=False
    ):
        self.listOfRecordings.remove(recordingObjectToRemove)
        if updateSavedFileIfAlreadySaved and self.savedFilePathName:
            saveDir = os.path.dirname(self.savedFilePathName)
            self.saveToFile(saveDir)

    def getListOfRecordingFileNames(self):
        listOfRecordingFileNames = [R.recordingFilePath for R in self.listOfRecordings]
        return listOfRecordingFileNames

    def getListOfScopeRuns(self):
        logging.debug("Session.getListOfScopeRuns()")
        listOfScopeRunsForSession = []
        for R in self.listOfRecordings:
            for S in R.listOfScopeRuns:
                listOfScopeRunsForSession.append(S)
        logging.debug(f"  {len(listOfScopeRunsForSession)} runs found in total")
        return listOfScopeRunsForSession


class Recording(object):
    def __init__(self, parentSessionObject, recordingFilePath, listOfScopeRuns=()):
        logging.debug("Recording object init()")
        self.parentSession = parentSessionObject
        self.recordingFilePath = (
            recordingFilePath  # should include full path and file name
        )
        self.listOfScopeRuns = listOfScopeRuns
        self.transformsNames = None
        self.transformsList = None

    def processRecordingToScopeRuns(
        self,
        sceneLeafTransformNode,
        progressObj: "ProgressObj",
        zoneTuples: Tuple[Tuple],
        segmentationNode,
        airwayZoneSegmentName="airwayZone",
    ):
        logging.debug("Recording.processRecordingToScopeRuns()")
        (
            self.transformsList,
            self.transformsNames,
        ) = gatherTransformsFromTransformHierarchy(
            leafTransformNode=sceneLeafTransformNode
        )
        scopeRuns = Recording.processRecordingFileToScopeRuns(
            self.recordingFilePath,
            sceneLeafTransformNode,
            segmentationNode,
            airwayZoneSegmentName,
        )
        for scopeRun in scopeRuns:
            scopeRun.setParentRecordingObject(self)
            try:
                scopeRun.userName = self.parentSession.userDataDict["userName"]
            except:
                # OK to fail silently here, I think, just means no parent session
                # or no userName
                pass
        self.listOfScopeRuns = scopeRuns
        # Analyze each run
        self.analyzeScopeRuns(progressObj, zoneTuples)

    def analyzeScopeRuns(self, progressObj, zoneTuples):
        """Run quantitative analysis on each scope run"""
        # NOTE: the zone analysis currently requires that the Slicer
        # scene is loaded and the ZONE_TUPLES constants are current. Probably
        # should refactor so these are inputs, maybe guidelet widget or logic
        # property?
        for scopeRun in self.listOfScopeRuns:
            scopeRun.analyze(progressObj, zoneTuples)

    def getTransformsInfoString(self):
        transform_names_str = json.dumps(self.transformsNames)
        transform_arrays_str = json.dumps([t.tolist() for t in self.transformsList])
        transformsInfoString = "\n".join([transform_names_str, transform_arrays_str])
        return transformsInfoString

    @classmethod
    def processRecordingFileToScopeRuns(
        cls,
        recordingFilePath,
        sceneLeafTransformNode,
        segmentationNode,
        airwayZoneSegmentName="airwayZone",
    ) -> List["ScopeRun"]:
        logging.debug("Recording.processRecordingFileToScopeRuns()")
        logging.info(
            f"  Processing {recordingFilePath}...\n   Using {sceneLeafTransformNode.GetName()} as transform leaf\n   Using {segmentationNode.GetName()} as segmentation\n"
        )
        (
            timeStamps,
            headSensorTransforms,
            scopeSensorTransforms,
        ) = cls.import_tracker_recording(recordingFilePath)
        transformsList, transformNames = gatherTransformsFromTransformHierarchy(
            leafTransformNode=sceneLeafTransformNode
        )
        # TODO: Verify that transformNames[1] looks like it's the dynamic head sensor and [2] looks like the dynamic scope sensor
        # Replace the single transform matrix arrays in transformsList with the full set from
        # the loaded file for both the head sensor and scope sensor
        HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 1
        SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY = 2
        # Replace the single transform at each appropriate location with the full sequence of
        # sensor transforms loaded from the recording file
        transformsList[HEAD_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = (
            headSensorTransforms
        )
        transformsList[SCOPE_SENSOR_TRANSFORM_POSITION_IN_HIERARCHY] = (
            scopeSensorTransforms
        )
        # Find sequence of positions/orientations using the t
        positions, orientationsZ, orientationsX = positions_from_transform_hierarchy(
            transformsList
        )
        # Break this sequence into separate runs
        runsData = identifyTrackingRunsFromRawPath(
            positions, segmentationNode, airwayZoneSegmentName
        )
        # runsData is a list, where each element is a list of indices associtated with a single run
        # indices are for timeStamps, positions, and orientations
        scopeRuns = []
        for runData in runsData:
            runData = np.array(runData)
            runPositions = positions[runData, :]
            runOrientationsZ = orientationsZ[runData, :]
            runOrientationsX = orientationsX[runData, :]
            runTimeStamps = timeStamps[runData]  # timeStamps is 1-D
            scopeRun = ScopeRun(
                None, runTimeStamps, runPositions, runOrientationsZ, runOrientationsX
            )
            scopeRuns.append(scopeRun)
        return scopeRuns

    @classmethod
    def import_tracker_recording(cls, mha_file_path):
        """Import the sequence of transforms stored in one of the guidelet mhd files."""
        # Can be used as Recording.import_tracker_recording()
        logging.debug(f"Recording.import_tracker_recording({mha_file_path})")
        logging.debug(f"  Opening file: {mha_file_path} ...")
        # Just parse far enough to get number of timesteps
        numSeqFrames = 0
        with open(mha_file_path) as f:
            for line in f:
                if line.startswith("DimSize"):
                    numSeqFrames = int(line.rstrip().split()[-1])
                    logging.debug(
                        f"  Found DimSize line: {numSeqFrames} time steps in file"
                    )
                    break
        if numSeqFrames == 0:
            raise (
                Exception(
                    "DimSize line not found in mha file processing, cannot determine number of processed time steps."
                )
            )
        # Preallocate transform arrays
        timeStamps = np.zeros((numSeqFrames))
        headSensorTransforms = np.zeros((4, 4, numSeqFrames))
        scopeSensorTransforms = np.zeros((4, 4, numSeqFrames))
        # Sequence line info pattern
        transformLinePatt = re.compile(
            r"Seq_Frame(?P<frameNumber>\d\d\d\d)_(?P<transformName>\w+) =(?P<matrix>( -?\d+(\.\d+)?([Ee][+-]?\d+)?){16})"
        )
        timeStampLinePatt = re.compile(
            r"Seq_Frame(?P<frameNumber>\d\d\d\d)_Timestamp = (?P<timeStamp>-?\d+(\.\d+)?)"
        )
        # Read the file line by line
        with open(mha_file_path) as f:
            keepgoing = True
            for line in f:
                if m := transformLinePatt.search(line):
                    # Process the result
                    groupDict = m.groupdict()
                    # logging.debug(groupDict.__str__())
                    frameNum = int(groupDict["frameNumber"])
                    transformName = groupDict["transformName"]
                    transformMatrixList = (
                        groupDict["matrix"].lstrip().split()
                    )  # row order
                    transformMatrix = np.array(transformMatrixList).reshape((4, 4))
                    # Store in arrays
                    if re.search("Head", transformName):
                        headSensorTransforms[:, :, frameNum] = np.linalg.inv(
                            transformMatrix.astype("float64")
                        )  # THESE NEED TO BE INVERTED!!!
                    elif re.search("Stylus", transformName):
                        scopeSensorTransforms[:, :, frameNum] = transformMatrix
                if m := timeStampLinePatt.search(line):
                    groupDict = m.groupdict()
                    frameNum = int(groupDict["frameNumber"])
                    timeStamp = float(groupDict["timeStamp"])
                    timeStamps[frameNum] = timeStamp
        return timeStamps, headSensorTransforms, scopeSensorTransforms


ADVANCING_PHASE_MAX_THRESH = 0.6  # Set this at epiglottis? 0.5? mid-trachea 0.65?
# OK, want to quantify times for phases which are not just the whole advancing time or whole withdrawal time
# It looks like ScopeRunPhase can likely already handle this, just by modifying the start and max frac threshes
#
POSTERIOR_NASOPHARYNX_PROGFRAC = 0.33  # Could plausibly be anywhere 0.31 to 0.4
EPIGLOTTIS_PROGFRAC = (
    0.47  # 0.46? 0.5 would mean definitely navigated past the epiglottis
)
START_RUN_PROGFRAC = -0.05  # change from 0.1 to capture nasal sill navigation time


class ScopeRun(object):
    def __init__(
        self, parentRecordingObject, timeStamps, positions, orientationsZ, orientationsX
    ):
        logging.debug("ScopeRun object init()")
        self.parentRecording = parentRecordingObject
        self.timeStamps = timeStamps
        self.positions = positions
        self.orientationsZ = orientationsZ
        self.orientationsX = orientationsX
        self.coneModel = None
        self.tubeModel = None
        self.contactModels = []
        self.userName = None

    def setParentRecordingObject(self, parentRecordingObject):
        logging.debug("ScopeRun.setParentRecordingObject()")
        self.parentRecording = parentRecordingObject

    def getSaveDataText(self):
        # organize data into saveable text format
        # Concatenate matrices to a 7-col array with timstamps, then positions, then orientations
        arr = np.concatenate(
            (
                self.timeStamps.reshape(len(self.timeStamps), 1),
                self.positions,
                self.orientationsZ,
                self.orientationsX,
            ),
            axis=1,
        )
        header_string = "JSON formatted list of run data. [timeStamp, pos_R, pos_A, pos_S, oriZ_R, oriZ_A, oriZ_S, oriX_R, oriX_A, oriX_S]"
        array_json = json.dumps(arr.tolist())
        saveDataText = "\n".join([header_string, array_json])

        # TODO make this a better format (csv?)
        # sections = []
        # sections.append('\nPostions:')
        # positions_string = np.array2string(self.positions)
        # sections.append(positions_string)
        # sections.append('\nOrientations:')
        # ori_string = np.array2string(self.orientations)
        # sections.append(ori_string)
        # sections.append('\nTimeStamps:')
        # timeStampsCol = self.timeStamps.reshape((len(self.timeStamps), 1)) # reformat to one number per column
        # tstamp_string = np.array2string(timeStampsCol)
        # sections.append(tstamp_string)
        # saveDataText = "\n".join(sections)
        return saveDataText

    def analyze(self, progObj: "ProgressObj", zoneTuples: Tuple[Tuple]):
        """Run quant analysis of scope run"""
        self.progressFracs = progObj.findProgressFractionsForScopeRun(self)
        self.validate()
        if self.valid:
            # Continue analysis
            speed, velocity = calcVelocity(self.positions, self.timeStamps, 15, "n")
            anglesDeg, cats = findOriToVelocityAngles(self.orientationsZ, velocity)
            self.speeds = speed
            self.velocities = velocity
            self.anglesDeg = anglesDeg
            self.angleCats = cats  # categorization of  angle
            advPhase = ScopeRunPhase(
                self,
                advancingFlag=True,
                maxFracThresh=ADVANCING_PHASE_MAX_THRESH,
                startFracThresh=START_RUN_PROGFRAC,
            )
            nasalPhase = ScopeRunPhase(
                self,
                advancingFlag=True,
                maxFracThresh=POSTERIOR_NASOPHARYNX_PROGFRAC,
                startFracThresh=START_RUN_PROGFRAC,
            )
            pharynxPhase = ScopeRunPhase(
                self,
                advancingFlag=True,
                maxFracThresh=EPIGLOTTIS_PROGFRAC,
                startFracThresh=POSTERIOR_NASOPHARYNX_PROGFRAC,
            )
            # Withdrawal phase (just quantify time?)
            wdrPhase = ScopeRunPhase(
                self,
                advancingFlag=False,
                maxFracThresh=ADVANCING_PHASE_MAX_THRESH,
                startFracThresh=START_RUN_PROGFRAC,
            )
            phasesToAnalyze = [advPhase, nasalPhase, pharynxPhase]
            for phase in phasesToAnalyze:
                phase.runPauseAnalysis()
                phase.runZoneAnalysis(zoneTuples=zoneTuples)
            # Store results in scopeRun
            self.advPhase = advPhase
            self.nasalPhase = nasalPhase
            self.pharynxPhase = pharynxPhase
            self.wdrPhase = wdrPhase
            # Score
            self.calcScore()
        else:
            # invalid scope run, do we want to do any reporting or
            # messaging here?
            pass

    def generateAnalysisReportText(self):
        """Create analysis report (text)"""
        if not self.valid:
            # Abbreviated text
            txt = f"Not a valid scope run:\nMaximum progress fraction was {np.max(self.progressFracs):0.2f} (>0.5 required)\nStarting progress fraction {self.progressFracs[0]:0.2f} (<0.1 required)"
            return txt
        # Otherwise, valid run, make normal report
        lines = []
        indent = "   "
        lines.append(f"Advancing duration: {self.advPhase.duration() :0.1f} s")
        lines.append(
            f"{indent}Nasal Phase duration: {self.nasalPhase.duration() :0.1f} s"
        )
        lines.append(
            f"{indent}Pharyngeal Phase duration: {self.pharynxPhase.duration() :0.1f} s"
        )
        lines.append("Undesirable anatomical contacts (#, duration)")
        zaDict = self.advPhase.zoneAnalysisDict
        for zoneName, za in zaDict.items():
            nSounds = za.nSounds
            totalContactTime = za.totalContactDuration
            lines.append(
                f"{indent}{zoneName}: {nSounds} contact{'' if nSounds==1 else 's'}, {totalContactTime:0.1f} s total contact time"
            )
        lines.append(f"Withdrawal duration: {self.wdrPhase.duration() :0.1f} s")
        # Assemble
        txt = "\n".join(lines)
        return txt

    def calcScore(self, minWithdrawalTimeSec=1, maxWithdrawalTimeSec=5):
        """Calculate score for leaderboard.
        Basic idea for scoring:
        10x adv phase time + penalties for contacts, contact duration, and
        too fast or too slow withdrawal.
        """
        if not self.valid:
            return None, None
        # Penalty weights
        timeWeight = 10  # points per second
        penaltyPerContact = 2 * timeWeight  # 2-second penalty for contact
        contactTimeWeight = 2 * timeWeight  # contact time counts triple
        minWithdrawalTime = minWithdrawalTimeSec  # sec
        tooFastWdrPenalty = 2 * timeWeight  # 2-second penalty for yanking
        maxWithdrawalTime = maxWithdrawalTimeSec  # sec
        tooSlowWdrPenalty = 1 * timeWeight  # 1-second penalty for slow-dragging
        # Gather data
        advPhaseDuration = self.advPhase.duration()
        nContactsList = [za.nSounds for za in self.advPhase.zoneAnalysisDict.values()]
        nTotalContacts = int(np.sum(nContactsList))
        totalContactDurationList = [
            za.totalContactDuration for za in self.advPhase.zoneAnalysisDict.values()
        ]
        totalContactDuration = np.sum(totalContactDurationList)
        wdrDuration = self.wdrPhase.duration()
        wdrPenalty = 0  # no penalty if within range
        if wdrDuration < minWithdrawalTime:
            wdrPenalty = tooFastWdrPenalty
        elif wdrDuration > maxWithdrawalTime:
            wdrPenalty = tooSlowWdrPenalty
        score = (
            advPhaseDuration * timeWeight
            + penaltyPerContact * nTotalContacts
            + contactTimeWeight * totalContactDuration
            + wdrPenalty
        )
        flawlessFlag = (
            nTotalContacts == 0 and totalContactDuration == 0 and wdrPenalty == 0
        )
        score = np.round(score)
        scoreComponents = {  # quantity, weight
            "advancingTime": (advPhaseDuration, timeWeight),
            "contactCountPenalty": (nTotalContacts, penaltyPerContact),
            "contactTimePenalty": (totalContactDuration, contactTimeWeight),
            "withdrawalPenalty": (wdrDuration, wdrPenalty),  # time, penalty
            "flawlessFlag": flawlessFlag,  # just bool
        }
        # Store results in scope run and also return them
        self.score = score
        self.scoreComponents = scoreComponents
        self.flawlessFlag = flawlessFlag
        return score, scoreComponents

    def validate(self, requiredMaxProgress=0.5, requiredStartProgress=0.1):
        """Mark scope run as valid if the run starts at or before the
        required start progress fraction, and if the maximum progress during
        the run is greater than or equal to the required maximum progress.
        Invalid runs will not be analyzed like valid runs (though some
        analysis of why they are invalid may be carried out and reported).
        """
        validFlag = True
        if np.max(self.progressFracs) < requiredMaxProgress:
            validFlag = False
        if self.progressFracs[0] > requiredStartProgress:
            validFlag = False
        self.valid = validFlag

    def saveToFile(self, saveDir):
        logging.debug("ScopeRun.saveToFile()")

    def createModelNodes(self, show=False):
        logging.debug("ScopeRun.createModelNode()")
        if self.positions is None or len(self.positions) < 1:
            raise (Exception("Can't create model node without positions!"))
        self.coneModel, self.tubeModel = modelNodesFromPositionsAndOrientations(
            self.positions, self.orientationsZ, scalars=None, sizeFactor=3.0
        )
        self.contactModels = self.createContactModels()
        if not show:
            self.hideModelNodes()

    def showModelNodes(self):
        logging.debug("ScopeRun.showModelNodes()")
        if self.coneModel:
            self.coneModel.GetDisplayNode().SetVisibility(True)
        if self.tubeModel:
           self.tubeModel.GetDisplayNode().SetVisibility(True)
        for contactModel in self.contactModels:
            contactModel.GetDisplayNode().SetVisibility(True)

    def hideModelNodes(self):
        logging.debug("ScopeRun.hideModelNodes()")
        if self.coneModel:
            self.coneModel.GetDisplayNode().SetVisibility(False)
        if self.tubeModel:
           self.tubeModel.GetDisplayNode().SetVisibility(False)
        for contactModel in self.contactModels:
            contactModel.GetDisplayNode().SetVisibility(False)

    def createContactModels(self):
        """Make a polydata point model of closest point contact locations for each
        zone, currently limited to advancing phase.  If zone analysis not present, 
        just return an empty list.
        """
        if not hasattr(self, 'advPhase'):
            logging.info("No advancing phase for current scope run, skipping contact model creation.")
            return []
        if not hasattr(self.advPhase, 'zoneAnalysisDict'):
            logging.info("No completed zone analysis for current scope run advancing phase, skipping contact model creation.")
            return []
        #
        zaDict = self.advPhase.zoneAnalysisDict
        contactModels = []
        for zoneName, za in zaDict.items():
            # Create contact location polydata model
            contactModel = za.createContactModel()
            if contactModel:
                contactModels.append(contactModel)
        return contactModels

        



class ProgressObj(object):
    def __init__(self, progressCurveNode, startFinishFiducialNode):
        startPointLoc = startFinishFiducialNode.GetNthControlPointPositionWorld(0)
        finishPointLoc = startFinishFiducialNode.GetNthControlPointPositionWorld(1)
        startIdx = findClosestControlPointIdx(startPointLoc, progressCurveNode)
        finishIdx = findClosestControlPointIdx(finishPointLoc, progressCurveNode)
        idxsRaw = np.array(range(progressCurveNode.GetNumberOfControlPoints()))
        progressFractions = (idxsRaw - startIdx) / (finishIdx - startIdx)
        self.curveNode = progressCurveNode
        self.fractions = progressFractions

    def findProgressFractionsForScopeRun(self, scopeRun):
        positions = scopeRun.positions
        progressFractions = np.zeros(positions.shape[0])  # pre-allocate with zeros
        for posIdx, pos in enumerate(positions):
            progressIdx = findClosestControlPointIdx(pos, self.curveNode)
            progressFractions[posIdx] = self.fractions[progressIdx]
        return progressFractions

    def __repr__(self):
        return f"ProgressObj object\n  curveNode: {self.curveNode.GetName()}\n  fractions: {self.fractions.shape} array "


class ScopeRunPhase(object):
    def __init__(
        self,
        parentScopeRun: ScopeRun,
        advancingFlag: bool,
        maxFracThresh=0.9,
        startFracThresh=0.1,
    ):
        self.parentScopeRun = parentScopeRun
        mask = np.full(parentScopeRun.timeStamps.shape, False)  # initialize
        if not parentScopeRun.valid:
            raise Exception(
                "Input ScopeRun object is not valid! Cancelling creation of ScopeRunPhase."
            )
        maxProgressIdx = np.argmax(parentScopeRun.progressFracs)
        if advancingFlag:
            self.phaseType = "advancing"
            mask = np.logical_and(
                parentScopeRun.progressFracs >= startFracThresh,
                parentScopeRun.progressFracs <= maxFracThresh,
            )
            mask[maxProgressIdx + 1 :] = False
            mask3 = np.column_stack((mask, mask, mask))
            numEntries = np.count_nonzero(mask)
        else:
            self.phaseType = "withdrawing"
            mask = np.logical_and(
                parentScopeRun.progressFracs >= startFracThresh,
                parentScopeRun.progressFracs <= maxFracThresh,
            )
            mask[:maxProgressIdx] = False
            mask3 = np.column_stack((mask, mask, mask))
            numEntries = np.count_nonzero(mask)
        self.timeStamps = parentScopeRun.timeStamps[mask] - parentScopeRun.timeStamps[0]
        self.progressFracs = parentScopeRun.progressFracs[mask]
        self.positions = parentScopeRun.positions[mask3].reshape(numEntries, 3)
        self.orientationsZ = parentScopeRun.orientationsZ[mask3].reshape(numEntries, 3)
        self.speeds = parentScopeRun.speeds[mask]
        self.anglesDeg = parentScopeRun.anglesDeg[mask]
        self.angleCats = parentScopeRun.angleCats[mask]

    def duration(self):
        return self.timeStamps[-1] - self.timeStamps[0]

    def runPauseAnalysis(
        self, pauseThreshSec=2, minProgVeloc=0.001, backtrackProgThresh=0.01
    ):
        """Carries out identification of pauses and backtrack events for
        this ScopeRun.  Pauses are times when the net advancement over
        at least pauseThreshSec falls behind the minProgVelocity advancement
        rate. Backtrack events are when the current progress fraction falls
        behind the maximum progress so far by at least backtrackProgThresh.
        """
        self.pauseIdxArray = findPauses2(
            self, pauseThreshSec, minProgVeloc, showFlag=False
        )
        self.nPauses = self.pauseIdxArray.shape[0]
        self.pauseDurations = findPauseDurations(self.pauseIdxArray, self.timeStamps)
        (
            self.nBacktrackEvents,
            self.backtrackEventDepths,
            self.backtrackEventMaxDepthIdxs,
        ) = findBacktrackEvents(self, backtrackProgThresh=backtrackProgThresh)

    def quickPlot(self):
        plt.plot(self.timeStamps - self.timeStamps[0], self.progressFracs)
        plt.xlabel("Time (sec)")
        plt.ylabel("Progress Fraction")
        plt.show()

    def runZoneAnalysis(self, zoneTuples):
        # Run an analysis of when the trajectory comes too close (in contact with)
        # zones representing places of particular irritation for patients.  When
        # sound production is on, touching these areas triggers an audible reaction
        # by playing a sound file.
        self.zoneAnalysisDict = dict()
        for displayName, zoneModelNode, triggerDist, soundDuration in zoneTuples:
            zoneModelName = zoneModelNode.GetName()
            self.zoneAnalysisDict[displayName] = ZoneAnalysisObj(
                self, zoneModelNode, triggerDist, soundDuration
            )


# defaultSoundDuration = 3.0  # sec
# ZONE_TUPLES = (
#    ("Septum", slicer.util.getNode("SeptumZoneTrimmed"), 2.00, defaultSoundDuration),
#    ("Nasopharynx", slicer.util.getNode("OuchZone2"), 3.00, defaultSoundDuration),
#    ("Epiglottis", slicer.util.getNode("GagZone"), 3.00, defaultSoundDuration),
#    ("Trachea", slicer.util.getNode("CoughZoneTrimmed"), 2.00, defaultSoundDuration),
# )


class ZoneAnalysisObj:
    def __init__(
        self, parent, zoneModelNode, triggerThreshDistanceMm, soundDelaySec=3.0
    ):
        self.parent = parent
        self.zoneModelNode = zoneModelNode
        self.triggerThreshDistanceMm = triggerThreshDistanceMm
        self.soundDelaySec = soundDelaySec
        self.runZoneAnalysis()

    def runZoneAnalysis(self):
        # Run (or re-run) zone analysis
        # For a sound trigger zone, we can associate every time point with a distance
        # to the closest point in the model, as well as a flag for whether this is
        # below the trigger threshold for contact/sound production.  For summarizing
        # purposes, we could record:
        #   the closest approach (minimum distance to each zone)
        #   total time spent in contact (below thresh)
        #   contact durations (contiguous times)
        #   number of sounds triggered (tricky because of sound duration dependence?)
        #   number of zones triggered (max 4?)
        # Might also be interesting to record the places that are scraped? (i.e. closest
        # points when below threshold).
        #self.rawZoneDists = distanceFromModel(self.parent.positions, self.zoneModelNode)
        self.rawZoneDists, closestPtArr = distancesAndClosestPointsFromModel(self.parent.positions, self.zoneModelNode)
        self.adjZoneDists = self.rawZoneDists - self.triggerThreshDistanceMm
        self.contactFlags = self.adjZoneDists <= 0
        self.contactPoints = closestPtArr[self.contactFlags]
        self.stepDurations = calcStepDurations(self.parent.timeStamps)
        self.contactDurations, self.contactIdxArray = calcZoneContactDurations(
            self.contactFlags, self.stepDurations
        )
        self.totalContactDuration = np.sum(self.contactDurations)
        self.minimumZoneDistance = np.min(self.adjZoneDists)
        # Sound trigger count (factor in forced delay while sound plays)
        self.soundTriggerIdxs = findSoundTriggerIdxs(
            self.parent.timeStamps, self.contactFlags
        )
        self.nSounds = len(self.soundTriggerIdxs)

    def getSummaryText(self, printMe=True):
        # Print a summary of the results of the analysis
        txtLines = [
            f"Total contact Duration: {self.totalContactDuration:0.2f} sec",
            f"Number of contacts: {len(self.contactDurations)}",
            f"Closest approach (adj): {np.min(self.adjZoneDists):0.2f} mm",
            f"Number of timesteps in contact: {np.sum(self.contactFlags)}",
        ]
        txt = "\n".join(txtLines)
        if printMe:
            print(txt)
        return txt

    def getCSVZoneVariables(self):
        # Return the set of variables which are to be included in the CSV file.
        # NOTE: THIS MUST BE COORDINATED WITH the generate_CSV function COLUMN HEADERS!!!
        # zoneVals = (('TotalContactTime', ' (s)'),
        #       ('ContactSoundCount', ''),
        #       ('ClosestApproach', ' (mm)'),
        #       )
        # Total contact time, number of sound triggers, closest approach
        outputs = (self.totalContactDuration, self.nSounds, self.minimumZoneDistance)
        return outputs
    
    def createContactModel(self, outputModelNode=None):
        if not hasattr(self, 'contactPoints'):
            return None
        
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        for contactPoint in self.contactPoints:
            pointID = points.InsertNextPoint(contactPoint)
            cellID = vertices.InsertNextCell(1)
            vertices.InsertCellPoint(pointID)
        pointsPolyData = vtk.vtkPolyData()
        pointsPolyData.SetPoints(points)
        pointsPolyData.SetVerts(vertices)

        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetRadius(2)
        glyphFilter = vtk.vtkGlyph3D()
        glyphFilter.SetSourceConnection(sphereSource.GetOutputPort())
        glyphFilter.SetInputData(pointsPolyData)

        if outputModelNode is None:
            modelName = f"{self.zoneModelNode.GetName()}_Contacts"
            outputModelNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLModelNode', modelName)
            outputModelNode.CreateDefaultDisplayNodes()
        # Connect to glyph filter
        outputModelNode.SetPolyDataConnection(glyphFilter.GetOutputPort())
        # Set color
        dn = outputModelNode.GetDisplayNode()
        dn.SetColor(1.0,0.0,0.0)
        return outputModelNode


class Leaderboard(object):
    """To keep track of scope run scores.  Should have the ability to display
    rankings restricting to unique usernames and including duplicate usernames.
    """

    def __init__(self, listOfScopeRuns=None, listOfEntries=None, savePath=None):
        if savePath is None:
            # use default name and save location
            timeStamp = time.strftime(r"%Y-%m-%d-%H%M%S")
            fileName = f"LeaderBoard_{timeStamp}.json"
            savePath = Path(slicer.app.temporaryPath, fileName)
        self.savePath = savePath
        self.listOfEntries: List[LeaderboardEntry] = listOfEntries or []
        if listOfScopeRuns is not None:
            for sr in listOfScopeRuns:
                self.addNewScopeRun(sr)
        self.sort()
        self.serialize()

    def addNewScopeRun(self, sr: ScopeRun):
        if not sr.valid:
            return
        entry = LeaderboardEntry(parentScopeRun=sr)
        self.addNewEntry(entry)

    def addNewEntry(self, entry: "LeaderboardEntry"):
        self.listOfEntries.append(entry)
        self.sort()

    def sort(self):
        self.listOfEntries.sort(key=lambda entry: entry.score)

    def getTopNResults(self, nResults: int = 10, uniqFlag: bool = True):
        """Get the top nResults scope runs"""
        topNList = []
        userNames = []
        for entry in self.listOfEntries:
            if uniqFlag and (entry.userName in userNames):
                # the better score came first, so safe to skip this one
                continue
            userNames.append(entry.userName)
            topNList.append(entry)
            if len(topNList) == nResults:
                break
        return topNList

    def display(self, nResults=10, uniqFlag=True, currentSr: Optional[ScopeRun] = None):
        """Show Qt table"""
        topNList = self.getTopNResults(nResults=nResults, uniqFlag=uniqFlag)
        show_leaderboard(topNList, currentSr)

    def serialize(self, filePath: Optional[Path] = None):
        """Save the current leaderboard entry data into a serializable text format."""
        if filePath is None:
            filePath = self.savePath
        if filePath.as_posix() == "":
            raise Exception("Tried to serialize to empty path")
        # Save
        txt = json.dumps(
            list([entry.serialize() for entry in self.listOfEntries]), indent=4
        )
        with open(filePath.as_posix(), "w") as f:
            f.write(txt)
        logging.debug(f"Successfully wrote leaderboard to '{filePath.as_posix()}'")

    @classmethod
    def deserialize(cls, filePath: Union[Path,str]):
        filePath = Path(filePath) # force to pathlib.Path object
        with filePath.open("r") as fp:
            entryDataList = json.load(fp)
        listOfEntries = []
        for entryDataStr in entryDataList:
            entry = LeaderboardEntry.deserialize(entryDataStr)
            listOfEntries.append(entry)
        return Leaderboard(listOfEntries=listOfEntries)


class LeaderboardEntry(object):
    def __init__(self, parentScopeRun: Optional[ScopeRun] = None):
        self.score: Optional[float] = None
        self.scoreComponents: Optional[dict] = None
        self.userName: Optional[str] = None
        self.flawlessFlag: Optional[bool] = True
        self.parentScopeRun: Optional[ScopeRun] = None  # This will not be serialized
        if parentScopeRun is not None:
            self.setParentScopeRun(parentScopeRun)

    def setParentScopeRun(self, parentScopeRun: ScopeRun):
        self.score = parentScopeRun.score
        self.scoreComponents = parentScopeRun.scoreComponents
        self.userName = parentScopeRun.userName
        self.flawlessFlag = parentScopeRun.flawlessFlag
        self.parentScopeRun = parentScopeRun

    def serialize(self) -> str:
        """
        Serializes the object to a JSON string, excluding the parentScopeRun.
        """
        # Create a dictionary with only the attributes you want to serialize
        data_to_serialize = {
            "score": self.score,
            "scoreComponents": self.scoreComponents,
            "userName": self.userName,
            "flawlessFlag": self.flawlessFlag,
        }
        return json.dumps(data_to_serialize, indent=4)

    @classmethod
    def deserialize(cls, json_string: str) -> "LeaderboardEntry":
        """
        Deserializes a JSON string into a new LeaderboardEntry object.
        The parentScopeRun will be None.
        """
        data = json.loads(json_string)

        # Create a new instance without a parentScopeRun
        new_entry = cls()

        # Populate the attributes from the loaded data
        new_entry.score = data.get("score")
        new_entry.scoreComponents = data.get("scoreComponents")
        new_entry.userName = data.get("userName")
        new_entry.flawlessFlag = data.get("flawlessFlag")

        return new_entry

    def __repr__(self):
        """A helper method for prettier printing."""
        return (
            f"LeaderboardEntry(userName='{self.userName}', score={self.score}, "
            f"flawless={self.flawlessFlag}, has_parent={self.parentScopeRun is not None})"
        )


class OLD_ScopeRun(object):
    # This is a copy of the old format for scopeRuns, before orientationX was being included
    def __init__(self, parentRecordingObject, timeStamps, positions, orientations):
        logging.debug("ScopeRun object init()")
        self.parentRecording = parentRecordingObject
        self.timeStamps = timeStamps
        self.positions = positions
        self.orientations = orientations
        self.coneModel = None
        self.tubeModel = None
        self.userName = None

    def setParentRecordingObject(self, parentRecordingObject):
        logging.debug("ScopeRun.setParentRecordingObject()")
        self.parentRecording = parentRecordingObject

    def getSaveDataText(self):
        # organize data into saveable text format
        # Concatenate matrices to a 7-col array with timstamps, then positions, then orientations
        arr = np.concatenate(
            (
                self.timeStamps.reshape(len(self.timeStamps), 1),
                self.positions,
                self.orientations,
            ),
            axis=1,
        )
        header_string = "JSON formatted list of run data. [timeStamp, pos_R, pos_A, pos_S, ori_R, ori_A, ori_S]"
        array_json = json.dumps(arr.tolist())
        saveDataText = "\n".join([header_string, array_json])

        # TODO make this a better format (csv?)
        # sections = []
        # sections.append('\nPostions:')
        # positions_string = np.array2string(self.positions)
        # sections.append(positions_string)
        # sections.append('\nOrientations:')
        # ori_string = np.array2string(self.orientations)
        # sections.append(ori_string)
        # sections.append('\nTimeStamps:')
        # timeStampsCol = self.timeStamps.reshape((len(self.timeStamps), 1)) # reformat to one number per column
        # tstamp_string = np.array2string(timeStampsCol)
        # sections.append(tstamp_string)
        # saveDataText = "\n".join(sections)
        return saveDataText

    def saveToFile(self, saveDir):
        logging.debug("ScopeRun.saveToFile()")

    def createModelNodes(self, show=False):
        logging.debug("ScopeRun.createModelNode()")
        if self.positions is None or len(self.positions) < 1:
            raise (Exception("Can't create model node without positions!"))
        self.coneModel, self.tubeModel = modelNodesFromPositionsAndOrientations(
            self.positions, self.orientations, scalars=None, sizeFactor=3.0
        )
        if not show:
            self.hideModelNodes()

    def showModelNodes(self):
        logging.debug("ScopeRun.showModelNodes()")
        self.coneModel.GetDisplayNode().SetVisibility(True)
        self.tubeModel.GetDisplayNode().SetVisibility(True)

    def hideModelNodes(self):
        logging.debug("ScopeRun.hideModelNodes()")
        self.coneModel.GetDisplayNode().SetVisibility(False)
        self.tubeModel.GetDisplayNode().SetVisibility(False)


## Helper functions not tied to a class or instance

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# MARK: FUNCTIONS
import pathlib


def getWaveFileDuration(waveFilePath: pathlib.Path):
    import wave

    with wave.open(waveFilePath.as_posix(), "r") as wavFile:
        frames = wavFile.getnframes()
        rate = wavFile.getframerate()
        duration = frames / float(rate)
    return duration


def calcVelocity(positions, timeStamps, halfWindow, halfWindowUnits="n"):
    # Force points to be in columns
    if positions.shape[0] != 3:
        positions = positions.T
    if halfWindowUnits == "n":
        # Integer number of sampling points
        posDiff = np.zeros(positions.shape)
        tsDiff = np.zeros(timeStamps.shape)
        posDiff[:, halfWindow:-halfWindow] = (
            positions[:, halfWindow * 2 :] - positions[:, : -halfWindow * 2]
        )
        tsDiff[halfWindow:-halfWindow] = (
            timeStamps[halfWindow * 2 :] - timeStamps[: -halfWindow * 2]
        )
        # Handle the points on either end whose window would extend outside the time window of the curve
        for idx in range(halfWindow):
            # Beginning
            posDiff[:, idx] = positions[:, idx + halfWindow] - positions[:, 0]
            tsDiff[idx] = timeStamps[idx + halfWindow] - timeStamps[0]
            # End
            posDiff[:, -1 - idx] = (
                positions[:, -1] - positions[:, -1 - idx - halfWindow]
            )
            tsDiff[-1 - idx] = timeStamps[-1] - timeStamps[-1 - idx - halfWindow]
        displacementMagnitude = np.linalg.norm(posDiff, axis=0)
        speed = np.divide(displacementMagnitude, tsDiff)
        velocityDirection = (
            posDiff / displacementMagnitude
        )  # normalize the vectors to have unit length
        velocityVector = (
            velocityDirection * speed
        )  # stretch vectors to have length of the speed
    elif halfWindowUnits == "sec":
        # Window in seconds
        tqForward = timeStamps + halfWindow
        tqBackward = timeStamps - halfWindow
        posForward = scipy.interpolate  # need to figure out matlab interp1 equivalent
    else:
        raise Exception('halfWindowUnits must be either "n" or "sec"')
    return speed, velocityVector


def findOriToVelocityAngles(orientation, velocity, thresholdAngleDeg=45):
    # Force vectors into the columns
    if orientation.shape[0] != 3:
        orientation = orientation.T
    if velocity.shape[0] != 3:
        velocity = velocity.T
    dotProductVector = np.einsum("ij,ij->j", orientation, velocity)
    # Find the angle between each orientation vector and each velocity vector
    anglesRad = np.arctan2(
        np.linalg.norm(np.cross(orientation, velocity, axis=0), axis=0),
        dotProductVector,
    )
    anglesDeg = 180.0 / np.pi * anglesRad
    # Categorize these based on the threshold angle into whether the tracked tip is advancing (category=1)
    # moving laterally (category=0) or withdrawing (category=-1)
    advancementCategories = np.zeros(anglesDeg.shape)
    advancementCategories[anglesDeg < thresholdAngleDeg] = 1  # advancing
    advancementCategories[
        (anglesDeg >= thresholdAngleDeg) & (anglesDeg <= (180 - thresholdAngleDeg))
    ] = 0  # lateral
    advancementCategories[anglesDeg > thresholdAngleDeg] = -1  # retreating/withdrawing
    return anglesDeg, advancementCategories


def showCat(cat, label=None):
    plt.plot(cat, label=label)
    plt.show()


def calcFrameRate(sr):
    # Time range
    timeRange = sr.timeStamps[-1] - sr.timeStamps[0]
    # Num time intervals
    numSteps = len(sr.timeStamps) - 1
    # Frame rate
    frameRate = numSteps / timeRange  # fps
    return frameRate


def createPhaseCurveNodes(phaseObjList):
    """Create a markupsCurveNode of positions for the phase, color it by phase type, and display it in Slicer"""
    advancingColor = [1, 0, 0]
    withdrawingColor = [0, 0, 1]
    markupsCurveList = []
    for listIdx, phaseObj in enumerate(phaseObjList):
        # Set color and short type tag based on phaseType
        if phaseObj.phaseType == "advancing":
            curveColor = advancingColor
            shortTag = "adv"
        elif phaseObj.phaseType == "withdrawing":
            curveColor = withdrawingColor
            shortTag = "wdr"
        else:
            raise Exception(f"Unknown phase type ({phaseObj.phaseType}) encountered!")
        # Construct node name based on user name and index into fullScopeRuns list (for easier cross-ref)
        nodeName = f"{phaseObj.parentScopeRun.userName}_{listIdx}_{shortTag}"
        # Create curve node
        markupsCurve = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", nodeName
        )
        markupsCurve.GetDisplayNode().SetVisibility(
            0
        )  # showing all of these slows stuff down a lot, hide by default
        slicer.util.updateMarkupsControlPointsFromArray(
            markupsCurve, phaseObj.positions
        )
        # Set color
        markupsCurve.GetDisplayNode().SetSelectedColor(curveColor)
        # Lock control points so that they can't accidentally be dragged
        for idx in range(markupsCurve.GetNumberOfControlPoints()):
            markupsCurve.SetNthControlPointLocked(idx, 1)
        # Store reference in phaseObj and return
        phaseObj.curveNode = markupsCurve
        markupsCurveList.append(markupsCurve)
    return markupsCurveList


def findClosestControlPointIdx(location, markupsNode):
    """Return the index of the closest control point to the supplied location on the supplied markupsNode"""
    controlPointsArray = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
    locationArray = np.array(location).reshape((1, 3))
    distances = np.linalg.norm(controlPointsArray - locationArray, axis=1)
    minIdx = np.argmin(distances)
    return minIdx


def calcProgressVelocity(phaseObj, halfWindowN=1):
    t = phaseObj.timeStamps
    pf = phaseObj.progressFracs
    pfDiff = np.zeros(pf.shape)
    tsDiff = np.zeros(t.shape)
    pfDiff[halfWindowN:-halfWindowN] = pf[halfWindowN * 2 :] - pf[: -halfWindowN * 2]
    tsDiff[halfWindowN:-halfWindowN] = t[halfWindowN * 2 :] - t[: -halfWindowN * 2]
    # handle points on either end by reducing the window size
    for idx in range(halfWindowN):
        # beginning
        pfDiff[idx] = pf[idx + halfWindowN] - pf[0]
        tsDiff[idx] = t[idx + halfWindowN] - t[0]
        # end
        pfDiff[-1 - idx] = pf[-1] - pf[-1 - idx - halfWindowN]
        tsDiff[-1 - idx] = t[-1] - t[-1 - idx - halfWindowN]
    progressVelocity = pfDiff / tsDiff
    return progressVelocity


def calcExpectedProgressCurve(timeStamps, progressFrac, minExpectedVelocity):
    expProgFrac = np.zeros(progressFrac.shape)

    for idx, pf in enumerate(progressFrac):
        if idx == 0:
            expProgFrac[idx] = pf
        else:
            expDelta = minExpectedVelocity * (timeStamps[idx] - timeStamps[idx - 1])
            expProgD = expProgFrac[idx - 1] + expDelta
            # Use the max of the expected progress or the actual progress
            expProgFrac[idx] = np.max([expProgD, pf])
    return expProgFrac


def calcMaxProgSoFarCurve(progressFracs):
    # Calculate a curve which is the cumulative maximal progress curve
    maxProgSoFar = np.zeros(progressFracs.shape)
    for idx in range(progressFracs.shape[0]):
        maxProgSoFar[idx] = np.max(progressFracs[: idx + 1])
    return maxProgSoFar


def calcProgressVelocity(phaseObj, halfWindowN=1):
    t = phaseObj.timeStamps
    pf = phaseObj.progressFracs
    pfDiff = np.zeros(pf.shape)
    tsDiff = np.zeros(t.shape)
    pfDiff[halfWindowN:-halfWindowN] = pf[halfWindowN * 2 :] - pf[: -halfWindowN * 2]
    tsDiff[halfWindowN:-halfWindowN] = t[halfWindowN * 2 :] - t[: -halfWindowN * 2]
    # handle points on either end by reducing the window size
    for idx in range(halfWindowN):
        # beginning
        pfDiff[idx] = pf[idx + halfWindowN] - pf[0]
        tsDiff[idx] = t[idx + halfWindowN] - t[0]
        # end
        pfDiff[-1 - idx] = pf[-1] - pf[-1 - idx - halfWindowN]
        tsDiff[-1 - idx] = t[-1] - t[-1 - idx - halfWindowN]
    progressVelocity = pfDiff / tsDiff
    return progressVelocity


def calcExpectedProgressCurve(timeStamps, progressFrac, minExpectedVelocity):
    expProgFrac = np.zeros(progressFrac.shape)

    for idx, pf in enumerate(progressFrac):
        if idx == 0:
            expProgFrac[idx] = pf
        else:
            expDelta = minExpectedVelocity * (timeStamps[idx] - timeStamps[idx - 1])
            expProgD = expProgFrac[idx - 1] + expDelta
            # Use the max of the expected progress or the actual progress
            expProgFrac[idx] = np.max([expProgD, pf])
    return expProgFrac


def calcMaxProgSoFarCurve(progressFracs):
    # Calculate a curve which is the cumulative maximal progress curve
    maxProgSoFar = np.zeros(progressFracs.shape)
    for idx in range(progressFracs.shape[0]):
        maxProgSoFar[idx] = np.max(progressFracs[: idx + 1])
    return maxProgSoFar


def findPauses2(phaseObj, pauseThreshSec=2, minExpectedVelocity=0.001, showFlag=False):
    """A pause is when the progress curve falls behind the minExpected velocity for
    at least pauseThreshSec seconds.
    """
    ts = phaseObj.timeStamps - phaseObj.timeStamps[0]
    pfc_orig = phaseObj.progressFracs
    if phaseObj.phaseType == "withdrawing":
        pfc = -1 * pfc_orig + 1
    elif phaseObj.phaseType == "advancing":
        pfc = pfc_orig
    mpc = calcMaxProgSoFarCurve(pfc)
    epc = calcExpectedProgressCurve(ts, pfc, minExpectedVelocity)
    curveDiff = epc - mpc
    allPauseMask = curveDiff > 0
    allPauseLabels, nLabels = scipy.ndimage.label(allPauseMask)
    # Throw out any pauses shorter than pauseThreshSec
    validPauses = []
    for pauseLabel in range(1, nLabels + 1):
        thisPauseIdxs = np.flatnonzero(allPauseLabels == pauseLabel)
        startIdx = thisPauseIdxs[0]
        lastIdx = thisPauseIdxs[-1]
        pauseDuration = ts[lastIdx] - ts[startIdx]
        if pauseDuration >= pauseThreshSec:
            validPauses.append([startIdx, lastIdx + 1])
    pauseIdxArray = np.array(validPauses, dtype=int)
    # Optionally show the identified pauses
    if showFlag:
        plt.plot(ts, pfc_orig, linewidth=0.5)
        for pauseStartIdx, pauseStopIdx in pauseIdxArray:
            plt.plot(
                ts[pauseStartIdx:pauseStopIdx],
                pfc_orig[pauseStartIdx:pauseStopIdx],
                linewidth=2,
            )
        plt.show()
    return pauseIdxArray


# Idea for findPauses3: instead of the max progress so far having infinite backwards memory,
# change this to a time window, say the maximum value in the last 5 sec. That is a way
# To avoid the excessive hill-climing after a long pause.


def findBacktrackEvents(phaseObj, backtrackProgThresh=0.01):
    pfc_orig = phaseObj.progressFracs
    if phaseObj.phaseType == "advancing":
        pfc = pfc_orig
    elif phaseObj.phaseType == "withdrawing":
        # Flip and shift
        pfc = -1 * pfc_orig + 1
    mpc = calcMaxProgSoFarCurve(pfc)
    curveDiff = mpc - pfc
    mask = curveDiff > backtrackProgThresh
    labelMap, nBacktrackEvents = scipy.ndimage.label(mask)
    # For each label, find the point of maximum backtracking
    eventDepths = np.zeros((nBacktrackEvents))
    eventMaxDepthIdxs = np.zeros((nBacktrackEvents), dtype=int)
    for eventIdx, label in enumerate(range(1, nBacktrackEvents + 1)):
        labelMask = labelMap == label
        labelMaskIdxs = np.flatnonzero(labelMask)
        depths = curveDiff[labelMask]
        maxDepth = np.max(depths)
        maxDepthIdxIntoLabelMask = np.argmax(depths)
        labelMaskIdxs = np.flatnonzero(labelMask)
        maxDepthIdx = labelMaskIdxs[maxDepthIdxIntoLabelMask]
        eventDepths[eventIdx] = maxDepth
        eventMaxDepthIdxs[eventIdx] = maxDepthIdx
    return nBacktrackEvents, eventDepths, eventMaxDepthIdxs


def addGridLines():
    # experimenting with adding background grid to plots
    plt.grid(which="major")  # both')
    # plt.grid(which='minor',linewidth=0.25)
    plt.grid(which="major", linewidth=0.5)
    xlimits = plt.xlimits
    # plt.minorticks_on()


def findPauseDurations(pauseIdxArray, ts):
    pauseDurations = np.zeros((pauseIdxArray.shape[0]))
    for idx, (start, stop) in enumerate(pauseIdxArray):
        pauseDurations[idx] = ts[stop - 1] - ts[start]
    return pauseDurations


def fancyPlot(sr, idx=None):
    # Show full curve with 0.5 thickness
    t = sr.timeStamps
    pf = sr.progressFracs
    plt.plot(t - t[0], pf, linewidth=0.5)
    # Show phases as thicker parts of curve
    advPhase = ScopeRunPhase(sr, True)
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    plt.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color
    #
    wPhase = ScopeRunPhase(sr, False)
    tw = wPhase.timeStamps
    pfw = wPhase.progressFracs
    plt.plot(tw, pfw, color=colors[0])
    # Show pauses in color overlays
    advPhase.runPauseAnalysis()
    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        plt.plot(
            ta[pauseStartIdx:pauseStopIdx],
            pfa[pauseStartIdx:pauseStopIdx],
            linewidth=2.5,
        )
    wPhase.runPauseAnalysis()
    for pauseStartIdx, pauseStopIdx in wPhase.pauseIdxArray:
        plt.plot(
            tw[pauseStartIdx:pauseStopIdx],
            pfw[pauseStartIdx:pauseStopIdx],
            linewidth=2.5,
        )
    # Add backtracking event points
    plt.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs],
        linestyle="",
        marker="o",
        markerfacecolor="None",
        color=colors[9],
    )
    plt.plot(
        tw[wPhase.backtrackEventMaxDepthIdxs],
        pfw[wPhase.backtrackEventMaxDepthIdxs],
        linestyle="",
        marker="o",
        markerfacecolor="None",
        color=colors[9],
    )

    # Grid
    addGridLines()
    # Title
    plt.title(f"{sr.userName}{' (Run %i)'%idx if idx else ''}")
    plt.xlabel("Time (s)")
    plt.ylabel("Progress Fraction")
    return plt


def fancyPlot2(sr, idx=None):
    """This is an updated version to more closely match the look of 2024 bootcamp figures"""
    # Show full curve with 0.5 thickness
    t = sr.timeStamps
    pf = sr.progressFracs
    plt.plot(t - t[0], pf, linewidth=0.5)
    # Show phases as thicker parts of curve
    advPhase = ScopeRunPhase(sr, True)
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    plt.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color
    #
    wPhase = ScopeRunPhase(sr, False)
    tw = wPhase.timeStamps
    pfw = wPhase.progressFracs
    plt.plot(tw, pfw, color=colors[0])
    # Show pauses in color overlays
    pauseLineWidth = 2.0
    pauseColor = "gray"
    pauseOffset = 0.05  # amount to shift the pause line from the progress fraction line
    advPhase.runPauseAnalysis()
    wPhase.runPauseAnalysis()

    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        pauseTs = ta[pauseStartIdx:pauseStopIdx]
        pauseY = pfa[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    for pauseStartIdx, pauseStopIdx in wPhase.pauseIdxArray:
        pauseTs = tw[pauseStartIdx:pauseStopIdx]
        pauseY = pfw[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    # Add backtracking event points
    plt.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs],
        linestyle="",
        marker="v",
        markerfacecolor="None",
        color=colors[9],
    )
    plt.plot(
        tw[wPhase.backtrackEventMaxDepthIdxs],
        pfw[wPhase.backtrackEventMaxDepthIdxs],
        linestyle="",
        marker="v",
        markerfacecolor="None",
        color=colors[9],
    )

    # Grid
    addGridLines()
    # Title
    plt.title(f"{sr.userName}{' (Run %i)'%idx if idx else ''}")
    plt.xlabel("Time (s)")
    plt.ylabel("Progress Fraction")
    return plt


def fancyPlot6(sr: ScopeRun, savePath=None, axTitleStr=""):
    """New version for 2025 bootcamp.  Uses the fact that the phases have
    already been calculated.
    """
    t = sr.timeStamps
    pf = sr.progressFracs
    fig, ax = plt.subplots()
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    # Show full curve with 0.5 thickness
    ax.plot(t - t[0], pf, linewidth=0.5, color=colors[0])

    # Show phases as thicker parts of curve
    advPhase = sr.advPhase
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    ax.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color
    wdrPhase = sr.wdrPhase
    tw = wdrPhase.timeStamps
    pfw = wdrPhase.progressFracs
    ax.plot(tw, pfw, color=colors[0], linewidth=1.0)

    # Show pauses in color overlays
    pauseLineWidth = 2.0
    pauseColor = "gray"
    pauseOffset = 0.05  # amount to shift the pause line from the progress fraction line
    zoneColors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0))
    zLineWidth = 3.0
    zShadeBuf = 0.2
    zSoundOpacity = 0.3
    contactStartMarkerSize = 5
    zSoundDuration = 3  # TODO propagate the actual value in here instead of hard coding
    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        pauseTs = ta[pauseStartIdx:pauseStopIdx]
        pauseY = pfa[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        ax.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    # Add backtracking event points
    backtrackMarkerColor = "red"
    backtrackOffset = 0.05
    ax.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )
    # Zone Contacts
    for zoneIdx, (zoneName, za) in enumerate(advPhase.zoneAnalysisDict.items()):
        for zStartIdx, zStopIdx in za.contactIdxArray:
            ax.plot(
                ta[zStartIdx:zStopIdx],
                pfa[zStartIdx:zStopIdx],
                linewidth=zLineWidth,
                color=zoneColors[zoneIdx],
            )
            # Add marker at start of contact, with filled middle and black edge
            ax.plot(
                ta[zStartIdx],
                pfa[zStartIdx],
                marker="d",
                markerfacecolor=zoneColors[zoneIdx],
                markeredgecolor="k",
                markersize=contactStartMarkerSize,
            )

        for soundStartIdx in za.soundTriggerIdxs:
            startTime = ta[soundStartIdx]
            endTime = startTime + zSoundDuration
            startProgFrac = pfa[soundStartIdx]
            # ax.fill_between([startTime,endTime],0,1, color=zoneColors[zoneIdx], alpha=0.5,transform=ax.get_xaxis_transform())
            ax.fill_between(
                [startTime, endTime],
                startProgFrac - zShadeBuf,
                startProgFrac + zShadeBuf,
                color=zoneColors[zoneIdx],
                alpha=zSoundOpacity,
            )

    # Limits
    # ax.set_xlim(0, 51)
    # ax.set_ylim(0, 0.55)
    # Grid
    # experimenting with adding background grid to plots
    ax.grid(which="major")  # both')
    # plt.grid(which='minor',linewidth=0.25)
    ax.grid(which="major", linewidth=0.5)
    xlimits = ax.get_xlim()
    xGridStepSize = 10
    ax.set_xticks(np.arange(0, xlimits[1], xGridStepSize))
    yGridStepSize = 0.1
    # ylimits = ax.get_ylim()
    # ax.set_yticks(np.arange(0, ylimits[1] + 0.01, yGridStepSize))
    ax.set_yticks(np.arange(0, 1.0 + 0.01, yGridStepSize))
    # Title
    if axTitleStr:
        ax.set_title(axTitleStr)
        # ax.set_title(f"{sr.userName}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Progress Fraction")
    # Add 2nd y axis on the right with labeled landmarks
    atickLabels = ["Nasal Sill", "Choanae", "Epiglottis", "Carina"]
    aticks = [0.0, POSTERIOR_NASOPHARYNX_PROGFRAC, EPIGLOTTIS_PROGFRAC, 1.0]
    ax2 = ax.twinx()
    ax2.set_yticks(aticks)
    ax2.set_yticklabels(atickLabels)
    ax2.set_ylabel("Anatomical Progress Landmarks")
    ax.set_ybound(upper=1.0)
    ax2.set_ylim(ax.get_ylim())

    fig.set_tight_layout(True)

    if savePath:
        fig.savefig(savePath)
        plt.close(fig)
    return fig


def fancyPlot3(sr, idx=None):
    """For figures for Anna with changes from fancyPlot2 to show backtracking triangles
    in red and make them larger.
    """
    # Show full curve with 0.5 thickness
    t = sr.timeStamps
    pf = sr.progressFracs
    plt.plot(t - t[0], pf, linewidth=0.5)
    # Show phases as thicker parts of curve
    advPhase = ScopeRunPhase(sr, True)
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    plt.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color
    #
    wPhase = ScopeRunPhase(sr, False)
    tw = wPhase.timeStamps
    pfw = wPhase.progressFracs
    plt.plot(tw, pfw, color=colors[0])
    # Show pauses in color overlays
    pauseLineWidth = 2.0
    pauseColor = "gray"
    pauseOffset = 0.05  # amount to shift the pause line from the progress fraction line
    advPhase.runPauseAnalysis()
    wPhase.runPauseAnalysis()

    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        pauseTs = ta[pauseStartIdx:pauseStopIdx]
        pauseY = pfa[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    for pauseStartIdx, pauseStopIdx in wPhase.pauseIdxArray:
        pauseTs = tw[pauseStartIdx:pauseStopIdx]
        pauseY = pfw[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    # Add backtracking event points
    backtrackMarkerColor = "red"
    backtrackOffset = 0.05
    plt.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )
    plt.plot(
        tw[wPhase.backtrackEventMaxDepthIdxs],
        pfw[wPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )

    # Grid
    # plt.grid(which='minor',linewidth=0.25)
    plt.grid(which="major", linewidth=0.5)
    xlimits = plt.xlim()
    xGridStepSize = 10
    plt.xticks(np.arange(0, xlimits[1], xGridStepSize))
    # ax.set_xticks(np.arange(0, xlimits[1], xGridStepSize))
    yGridStepSize = 0.1
    ylimits = plt.ylim()  # ax.set_ylim()
    plt.yticks(np.arange(0, ylimits[1] + 0.01, yGridStepSize))
    # ax.set_yticks(np.arange(0, ylimits[1] + 0.01, yGridStepSize))

    # Title
    plt.title(f"{sr.userName}{' (Run %i)'%idx if idx else ''}")
    plt.xlabel("Time (s)")
    plt.ylabel("Progress Fraction")
    return plt


def fancyPlot4(sr, idx=None):
    """Final figures for Anna with changes from fancyPlot3 to adjust
    ticks, force same xlim extent.
    """
    # Show full curve with 0.5 thickness
    t = sr.timeStamps
    pf = sr.progressFracs
    plt.plot(t - t[0], pf, linewidth=0.5)
    # Show phases as thicker parts of curve
    advPhase = ScopeRunPhase(sr, True)
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    plt.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color
    #
    wPhase = ScopeRunPhase(sr, False)
    tw = wPhase.timeStamps
    pfw = wPhase.progressFracs
    plt.plot(tw, pfw, color=colors[0])
    # Show pauses in color overlays
    pauseLineWidth = 2.0
    pauseColor = "gray"
    pauseOffset = 0.05  # amount to shift the pause line from the progress fraction line
    advPhase.runPauseAnalysis()
    wPhase.runPauseAnalysis()

    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        pauseTs = ta[pauseStartIdx:pauseStopIdx]
        pauseY = pfa[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    for pauseStartIdx, pauseStopIdx in wPhase.pauseIdxArray:
        pauseTs = tw[pauseStartIdx:pauseStopIdx]
        pauseY = pfw[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        plt.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    # Add backtracking event points
    backtrackMarkerColor = "red"
    backtrackOffset = 0.05
    plt.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )
    plt.plot(
        tw[wPhase.backtrackEventMaxDepthIdxs],
        pfw[wPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )

    # Limits
    plt.xlim(0, 125)
    plt.ylim(0, 1)
    # Grid
    # experimenting with adding background grid to plots
    plt.grid(which="major")  # both')
    # plt.grid(which='minor',linewidth=0.25)
    plt.grid(which="major", linewidth=0.5)
    xlimits = plt.xlim()
    xGridStepSize = 10
    plt.xticks(np.arange(0, xlimits[1], xGridStepSize))
    yGridStepSize = 0.1
    ylimits = plt.ylim()
    plt.yticks(np.arange(0, ylimits[1] + 0.01, yGridStepSize))
    # Title
    plt.title(f"{sr.userName}{' (Run %i)'%idx if idx else ''}")
    plt.xlabel("Time (s)")
    plt.ylabel("Progress Fraction")
    return plt


r"""To make the poster plot for Anna's poster, she chose the two subjects she
wanted to show, then I manually pulled code from the bootcamp2024.py script
and the TrackerCurveAnalysisExploration.ipynb to load the session files
and construct the trajectories. (Load into ScopeRun, generate new
ScopeRunPhase, do most of the steps in the early bootcamp2024.py script
to generate progress fractions and advancing phase with custom endpoints,
then show trajectory via updateMarkupsControlPointsFromArray for a
new curve markup node and sr.positions as the array. Finally, adjust 
display properties and hide all points except every 15th one (frame rate
is relatively consistent at 15 fps). Saved in 
C:\Users\mike.bindschadler@seattlechildrens.org\OneDrive - SCH\Airway4D\Temp\TrackerFiles
\2024\ToShare\AfterBootCampSceneSave\AnnaPosterTrajectories.mrb)
"""


def fancyPlot5(advPhase, savePath=None):
    """Final figures for Anna's poster with changes from fancyPlot4 to adjust
    ticks, force same xlim extent.
    """
    # Show full curve with 0.5 thickness
    # t = sr.timeStamps
    # pf = sr.progressFracs
    # plt.plot(t-t[0], pf, linewidth=0.5)
    # Show phases as thicker parts of curve
    # advPhase = ScopeRunPhase(sr, True)
    ta = advPhase.timeStamps
    pfa = advPhase.progressFracs
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    fig, ax = plt.subplots()
    ax.plot(ta, pfa, color=colors[0], linewidth=1.5)  # match the color

    # Show pauses in color overlays
    pauseLineWidth = 2.0
    pauseColor = "gray"
    pauseOffset = 0.05  # amount to shift the pause line from the progress fraction line
    advPhase.runPauseAnalysis()

    for pauseStartIdx, pauseStopIdx in advPhase.pauseIdxArray:
        pauseTs = ta[pauseStartIdx:pauseStopIdx]
        pauseY = pfa[pauseStartIdx] + pauseOffset
        pauseYs = np.full(pauseTs.shape, pauseY)
        ax.plot(pauseTs, pauseYs, linewidth=pauseLineWidth, color=pauseColor)
    # Add backtracking event points
    backtrackMarkerColor = "red"
    backtrackOffset = 0.05
    ax.plot(
        ta[advPhase.backtrackEventMaxDepthIdxs],
        pfa[advPhase.backtrackEventMaxDepthIdxs] + backtrackOffset,
        linestyle="",
        marker="v",
        markerfacecolor=backtrackMarkerColor,
        color=backtrackMarkerColor,
        markersize=5,
    )

    # Limits
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 0.55)
    # Grid
    # experimenting with adding background grid to plots
    ax.grid(which="major")  # both')
    # plt.grid(which='minor',linewidth=0.25)
    ax.grid(which="major", linewidth=0.5)
    xlimits = ax.get_xlim()
    xGridStepSize = 10
    ax.set_xticks(np.arange(0, xlimits[1], xGridStepSize))
    yGridStepSize = 0.1
    ylimits = ax.set_ylim()
    ax.set_yticks(np.arange(0, ylimits[1] + 0.01, yGridStepSize))
    # Title
    # ax.set_title(f"{sr.userName}{' (Run %i)'%idx if idx else ''}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Progress Fraction")
    if savePath:
        fig.savefig(savePath)
        plt.close(fig)
    return fig


def loadAllScopeRunsFromDirectory(dirName):
    """ """
    scopeRunList = []
    patt = re.compile("^Session.*[.]txt$")
    # Get list of files in that directory which start with "Session" and end in ".txt"
    files = [f for f in os.listdir(dirName) if patt.match(f)]
    for f in files:
        filePathName = os.path.join(dirName, f)
        fileScopeRunList = loadOnlyScopeRunsFromSessionFile(filePathName)
        scopeRunList.extend(fileScopeRunList)
    return scopeRunList


def loadOnlyScopeRunsFromSessionFile(filePathName):
    """ """
    scopeRunList = []
    scopeRunHeaderLinePatt = re.compile("JSON formatted list of run data")
    with open(filePathName, "r") as f:
        userName = f.readline().rstrip()
        for line in f:
            if scopeRunHeaderLinePatt.match(line):
                arr = np.array(json.loads(next(f)))
                timeStamps = arr[:, 0]
                positions = arr[:, 1:4]
                orientationsZ = arr[:, 4:7]
                orientationsX = arr[:, 7:]
                S = ScopeRun(None, timeStamps, positions, orientationsZ, orientationsX)
                S.userName = userName
                scopeRunList.append(S)
    return scopeRunList


def gatherTransformsFromTransformHierarchy(leafTransformNode):
    """Use the existing scene transform hierarchy to build a list of
    transformation matrices (called transformsList)
    above a given leaf transform node (i.e. the list returned will be all
    transforms above the leaf (inclusive))
    """
    logging.debug("HelperClasses.gatherTransformsFromTransformHierarchy()")
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
    transformList = [
        slicer.util.arrayFromTransformMatrix(tNode) for tNode in transformNodeList
    ]
    return transformList, transformNames


def positions_from_transform_hierarchy(transformsList):
    """Compute series of locations given a list of hierarchical transform matrices.
    transformsList[i] must either be a single 4x4 array or a 4x4xN array, where N is
    the number of time step frames. transformsList[i] is the parent transform to
    transformsList[i+1]
    """
    numFramesPer = np.zeros((len(transformsList)), dtype=int)
    for idx, tListItem in enumerate(transformsList):
        numDims = tListItem.ndim
        if numDims == 3:
            numFramesPer[idx] = tListItem.shape[2]
        elif numDims == 2:
            numFramesPer[idx] = 1
    # NumFramesPer elements should now all either equal 1 or the same number of frames
    numFrames = np.max(numFramesPer)
    assert np.all(
        (numFramesPer == 1) | (numFramesPer == numFrames)
    ), "All transforms supplied must have either a single frame or the same number of frames!"
    # Calculate positions from the sequence of transforms
    positions = np.zeros((numFrames, 3))
    orientationsZ = np.zeros((numFrames, 3))
    orientationsX = np.zeros((numFrames, 3))
    # origin = np.zeros((4))
    # origin[3] = 1  # homogenous coordinate for a point
    # forwardDirection = np.zeros((4))
    # forwardDirection[2] = 1  # [0,0,1,0]
    # forwardDirection[3] = 0  # homogenous coord for a vector
    concatTransforms = []
    for frameNum in range(numFrames):
        # Assemble the correct list of transforms for this frame
        currentTransformList = []
        for listIdx, transformsListItem in enumerate(transformsList):
            if numFramesPer[listIdx] == 1:
                curTransform = transformsListItem
            else:
                curTransform = transformsListItem[:, :, frameNum]
            currentTransformList.append(curTransform)
        # Apply tranforms in order to origin position
        concatTransform = np.linalg.multi_dot(currentTransformList)
        # Read position and orientation directions from concatenated transform
        currentPosition = concatTransform[0:3, 3]
        currentOrientationZ = concatTransform[0:3, 2]  # forward=+Z
        currentOrientationX = concatTransform[0:3, 1]  # normal (related to camera up)
        # Z = X x Y ; Z x X = Y, so
        # currentOrientationY = Z x X  (vector cross products)
        concatTransforms.append(concatTransform)
        #
        # currentPosition4 = concatTransform @ origin
        # currentOrientation4 = concatTransform @ forwardDirection
        positions[frameNum, :] = currentPosition[:]
        orientationsZ[frameNum, :] = currentOrientationZ[:]
        orientationsX[frameNum, :] = currentOrientationX[:]
    return positions, orientationsZ, orientationsX


def buildConcatTransform4x4FromPosOriZOriX(position, oriZ, oriX):
    """Rebuild the concatenated transform from a position, forward vector (z-axis), and
    the x transverse vector (x-axis).
    """
    # Normalize the input orientation vectors to unit length (just in case)
    oriZ = oriZ / np.linalg.norm(oriZ)
    oriX = oriX / np.linalg.norm(oriX)
    oriY = np.cross(oriZ, oriX)
    # Concatenated transform
    concatTransform = np.eye(4)
    concatTransform[0, 0:3] = oriX
    concatTransform[1, 0:3] = oriY
    concatTransform[2, 0:3] = oriZ
    concatTransform[3, 0:3] = position
    return concatTransform


def identifyTrackingRunsFromRawPath(
    positions, segmentationNode, airwayZoneSegmentName, minimumRunLengthPoints=30
):
    """Using a segmentation node which has a segment for the airway zone, combined
    with a set of sequentual positions, divide these positions in to "runs".
    A run is a (mostly) contiguous section of the position indices which includes at least one
    airwayZone point and all surrounding airwayZone points.
    Runs are merged if they have less than a 10 point gap between them (assumed due to
    aberrent sensor readings)
    """
    logging.debug("HelperClasses.identifyTrackingRunsFromRawPath()")
    runsData = []
    # points = slicer.util.arrayFromMarkupsControlPoints(markupsNode)
    points = positions
    nPoints = points.shape[0]
    segmentNames = getSegmentNamesAtRasPoint(segmentationNode, points)
    # Make a mask of airwayZone points
    airwayZoneMask = np.zeros((nPoints))
    for idx in range(nPoints):
        segNamesNow = segmentNames[idx]
        airwayZoneMask[idx] = 1 if airwayZoneSegmentName in segNamesNow else 0

    # If no airwayZone points, no runs
    if np.all(airwayZoneMask == 0):
        logging.info(
            f"No runs present because no points are inside {airwayZoneSegmentName} segment!"
        )
        return []

    while np.any(airwayZoneMask == 1):
        # Find first deep point index
        startIdx = np.argmax(airwayZoneMask)
        # Run forwards to find the last contiguous point which is still in zone
        stopIdx = startIdx + 1
        while True:
            if stopIdx == len(airwayZoneMask) or (airwayZoneMask[stopIdx] != 1):
                break
            else:
                # increment
                stopIdx = stopIdx + 1
        # Store data
        runData = list(range(startIdx, stopIdx))
        logging.debug(f"Run identified from index {startIdx} to {stopIdx}.")
        runsData.append(runData)
        # Clear this run from deepMask
        airwayZoneMask[startIdx:stopIdx] = 0
    # Merge any runs which are separated by only a short break (assume the gap is
    # due to an aberrent sensor reading)
    RUN_GAP_TOLERANCE = 10  # at normal sampling rates, this represents about 1/3 second
    runDataGaps = np.array(
        [runsData[idx + 1][0] - runsData[idx][-1] for idx in range(len(runsData) - 1)]
    )
    while np.any(runDataGaps <= RUN_GAP_TOLERANCE):
        # Merge the first one and then see if any remain
        firstGapBelowTolIdx = np.nonzero(runDataGaps <= RUN_GAP_TOLERANCE)[0][
            0
        ]  # np.nonzero result is for
        # 2D even if input is 1D, the first [0] is to get to the first dimension index answers, the second [0]
        # is because we want the first index in the first dimension
        runsData[firstGapBelowTolIdx] = [
            *runsData[firstGapBelowTolIdx],
            *runsData[firstGapBelowTolIdx + 1],
        ]
        del runsData[firstGapBelowTolIdx + 1]  # remove duplicate of merged section
        logging.debug(
            f"  Merged run {firstGapBelowTolIdx} and {firstGapBelowTolIdx+1} because gap was < {RUN_GAP_TOLERANCE} points!"
        )
        runDataGaps = np.array(
            [
                runsData[idx + 1][0] - runsData[idx][-1]
                for idx in range(len(runsData) - 1)
            ]
        )
    # Enforce minimum run lengths
    runLengths = np.array([len(runData) for runData in runsData])
    numTooShortRuns = np.count_nonzero(runLengths < minimumRunLengthPoints)
    if numTooShortRuns > 0:
        logging.debug(
            f"   Dropped {numTooShortRuns} for being less than {minimumRunLengthPoints} points long"
        )
    runsData = [
        runData for runData in runsData if len(runData) >= minimumRunLengthPoints
    ]

    return runsData


def getSegmentNamesAtRasPoint(
    segmentationNode,
    rasPoints=([0, 0, 0], [1, 1, 1]),
    includeHiddenSegments=True,
    sliceViewLabel="Green",
):
    """Returns names of segments at the rasPoint location.  If includeHiddenSegments is false (default is True)
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
    tempDisplayNode = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationDisplayNode"
    )
    if not includeHiddenSegments:
        # Copy everything (crucially, including current segment visibility settings)
        tempDisplayNode.Copy(segmentationNode.GetDisplayNode())
        # If this is not done, all segments are visible by default, and therefore will be included in the output
    tempDisplayNode.SetVisibility3D(0)  # no need to show in 3D
    tempDisplayNode.SetVisibility2D(1)
    tempDisplayNode.SetViewNodeIDs((sliceNode.GetID(),))
    tempDisplayNode.SetVisibility(1)
    segmentationNode.AddAndObserveDisplayNodeID(tempDisplayNode.GetID())

    segmentationsDisplayableManager = (
        sliceViewWidget.sliceView().displayableManagerByClassName(
            "vtkMRMLSegmentationsDisplayableManager2D"
        )
    )
    # Loop over ras points
    segmentNames = []
    nPoints = len(rasPoints)
    for pointIdx in range(nPoints):
        rasPoint = rasPoints[pointIdx]
        # Jump to slice containing query point
        sliceNode.JumpSliceByOffsetting(*rasPoint)
        # Get list of segment names at that point
        segmentIds = vtk.vtkStringArray()
        segmentationsDisplayableManager.GetVisibleSegmentsForPosition(
            rasPoint, tempDisplayNode, segmentIds
        )
        segmentNamesForCurrentPoint = [
            segmentationNode.GetSegmentation()
            .GetSegment(segmentIds.GetValue(idx))
            .GetName()
            for idx in range(segmentIds.GetNumberOfValues())
        ]
        segmentNames.append(segmentNamesForCurrentPoint)
    # Restore prior state
    sliceNode.SetSliceOffset(oldOffset)
    segmentationNode.RemoveNthDisplayNodeID(
        segmentationNode.GetNumberOfDisplayNodes() - 1
    )
    return segmentNames


def modelNodesFromPositionsAndOrientations(
    positions, orientations, scalars=None, sizeFactor=2.0
):
    """Create model node with polydata (for non-interactive version of markups)
    Positions will be centers of cones, cones will point the tip in the directions of orientations,
    cones will be sized individually by scalars and then uniformly multiplied by sizeFactor.
    Helpful links:
    https://www.dillonbhuff.com/?p=540
    https://vtk.org/doc/release/5.0/html/a01880.html#:~:text=vtkPolyData%20is%20a%20data%20object,also%20are%20represented.
    """
    # TODO: Add velocities (size), deviations from expert (color)?
    import numpy as np

    pointsArray = positions  # an nx3 numpy array
    numPoints = pointsArray.shape[0]
    # diffs = np.diff(pointsArray, axis=0)
    # Expand scalars to array if needed
    if scalars is None:
        scalars = np.ones(numPoints)
    elif isinstance(scalars, (list, tuple, np.ndarray)) and len(scalars) == 1:
        scalars = scalars[0] * np.ones(numPoints)
    else:
        # Single value not in a list
        scalars = scalars * np.ones(numPoints)

    # Create VTK arrays needed for model node
    points = vtk.vtkPoints()  # actual pointData locations
    vertices = (
        vtk.vtkCellArray()
    )  # handles vertex locations (in our case this will just be all the points, but note that
    # points could be a superset of vertices because points could include the endpoints of lines or corners of polygons
    # which don't themselves have to be in the list of vertices.  Vertices are what the vtkGlyph3D filter operates on
    lines = (
        vtk.vtkCellArray()
    )  # handles lines (and same type would handle polygons if those were being used)

    vectors = (
        vtk.vtkFloatArray()
    )  # this will affect glyph orientation and is going to be set to the vector to the next point
    vectors.SetNumberOfComponents(3)
    vectors.SetName("Directions")

    sizes = vtk.vtkFloatArray()  # this will go in scalars
    sizes.SetName("Sizes")
    # colors = vtk.vtkFloatArray()
    # colors.SetName("Colors"

    # Assemble arrays of values
    for point, orientation, scalar in zip(positions, orientations, scalars):
        pointID = points.InsertNextPoint(point)
        # Vertices
        cellID = vertices.InsertNextCell(
            1
        )  # allocates a next cell with space for one point ID (lines would have 2, triangles 3, polygons N)
        vertices.InsertCellPoint(
            pointID
        )  # fills the first (and only) slot for this cell
        # Vectors
        vectorID = vectors.InsertNextTuple(orientation)
        # Speed?? Could be calculated here and used to size the cones? TODO
        sizes.InsertNextValue(scalar * sizeFactor)
        if pointID != (numPoints - 1):
            # Add a line unless this is the very last point
            lines.InsertNextCell(2)
            # allocates a next cell with space for two pointIDs (the endpoints of the line)
            lines.InsertCellPoint(pointID)
            lines.InsertCellPoint(pointID + 1)
            # size = e[i] * sizeFactor
            # _ = sizes.InsertNextValue(size)
            # colorIdx = cmapIndices[i]
            # _ = colors.InsertNextValue(colorIdx)

    ## Create the vtkPolyData
    pointsPolyData = vtk.vtkPolyData()
    pointsPolyData.SetPoints(points)
    pointsPolyData.SetVerts(vertices)
    pointsPolyData.SetLines(lines)
    pointsData = pointsPolyData.GetPointData()
    _ = pointsData.SetScalars(
        sizes
    )  # scalars are literally used as size scale factors, I think
    _ = pointsData.SetVectors(vectors)
    # _ = pointsData.AddArray(colors)

    # sphere = vtk.vtkSphereSource()  # ConeSource()
    cone = vtk.vtkConeSource()
    cone.SetResolution(18)
    cone.SetRadius(0.30)

    linesPolyData = vtk.vtkPolyData()
    linesPolyData.SetPoints(points)
    linesPolyData.SetLines(lines)

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(linesPolyData)
    tubeFilter.SetRadius(0.25)  # tube radius mm
    tubeFilter.SetNumberOfSides(15)

    glyphFilter = vtk.vtkGlyph3D()
    glyphFilter.SetSourceConnection(cone.GetOutputPort())
    glyphFilter.SetInputData(pointsPolyData)

    coneModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "OriCones")
    coneModel.CreateDefaultDisplayNodes()
    # modelDisplay = coneModel.GetDisplayNode()
    # modelDisplay.SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')
    # modelDisplay.SetAndObserveColorNodeID(
    #    "vtkMRMLColorTableNodeFileColdToHotRainbow.txt"
    # )
    # Color table names can be found at https://apidocs.slicer.org/master/classvtkMRMLColorLogic.html
    # I found that changing the color using the Model display node GUI crashes slicer, I don't know why
    # modelDisplay.SetScalarVisibility(True)
    # modelDisplay.SetActiveScalarName("Colors")

    # Connect to glyph output
    coneModel.SetPolyDataConnection(glyphFilter.GetOutputPort())

    # Lines version
    tubeModel = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "Tube")
    tubeModel.CreateDefaultDisplayNodes()
    tubeModel.SetPolyDataConnection(tubeFilter.GetOutputPort())
    return coneModel, tubeModel


def distanceFromModel(positionArray, modelNode):
    # Transform model polydata to world coordinate system
    if modelNode.GetParentTransformNode():
        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            modelNode.GetParentTransformNode(), None, transformModelToWorld
        )
        polyTransformToWorld = vtk.vtkTransformPolyDataFilter()
        polyTransformToWorld.SetTransform(transformModelToWorld)
        polyTransformToWorld.SetInputData(modelNode.GetPolyData())
        polyTransformToWorld.Update()
        surface_World = polyTransformToWorld.GetOutput()
    else:
        surface_World = modelNode.GetPolyData()
    # Set up filter
    distanceFilter = vtk.vtkImplicitPolyDataDistance()
    distanceFilter.SetInput(surface_World)
    #
    distanceArr = np.zeros(positionArray.shape[0])
    for idx in range(positionArray.shape[0]):
        distanceArr[idx] = distanceFilter.EvaluateFunction(positionArray[idx, :])
        # note, if desired, the closest point on the model could also be returned
        # using EvaluateFunctionAndGetClosestPoint(pt, closestPt)
    return distanceArr

def distancesAndClosestPointsFromModel(positionArray, modelNode):
    # Transform model polydata to world coordinate system
    if modelNode.GetParentTransformNode():
        transformModelToWorld = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            modelNode.GetParentTransformNode(), None, transformModelToWorld
        )
        polyTransformToWorld = vtk.vtkTransformPolyDataFilter()
        polyTransformToWorld.SetTransform(transformModelToWorld)
        polyTransformToWorld.SetInputData(modelNode.GetPolyData())
        polyTransformToWorld.Update()
        surface_World = polyTransformToWorld.GetOutput()
    else:
        surface_World = modelNode.GetPolyData()
    # Set up filter
    distanceFilter = vtk.vtkImplicitPolyDataDistance()
    distanceFilter.SetInput(surface_World)
    #
    distanceArr = np.zeros(positionArray.shape[0])
    closestPtArr = np.zeros((positionArray.shape[0], 3))
    for idx in range(positionArray.shape[0]):
        closestPt = np.zeros(3)
        distanceArr[idx] = distanceFilter.EvaluateFunctionAndGetClosestPoint(positionArray[idx, :],closestPt)
        closestPtArr[idx,:] = closestPt
       
    return distanceArr, closestPtArr


def calcStepDurations(timeStamps):
    """Calculate step durations in time. The step duration is taken to be the
    time between the midpoints of the intervals befor and after a step timeStamp.
    For the first and last timeStamp, the duration is from the endpoint timeStamp
    to the adjacent midpoint (so typically half as long as other steps). This
    interval could be doubled if it would make more sense for the durations to be
    more comparable at the endpoints.
    """
    midTimes = (timeStamps[:-1] + timeStamps[1:]) / 2
    augMidTimes = np.hstack((timeStamps[0], midTimes, timeStamps[-1]))
    stepDurations = augMidTimes[1:] - augMidTimes[:-1]
    return stepDurations


def calcZoneContactDurations(contactFlags, stepDurations):
    contactsLabelMap, nContactEvents = scipy.ndimage.label(contactFlags)
    contactDurations = []
    contactStartIdxs = []
    contactEndIdxs = []
    for labelIdx in range(1, nContactEvents + 1):  # skip background label zero
        contactMask = contactsLabelMap == labelIdx
        contactDuration = np.sum(stepDurations[contactMask])
        contactIdxs = np.flatnonzero(contactMask)
        contactStartIdx = contactIdxs[0]
        contactEndIdx = contactIdxs[-1]
        # Store
        contactDurations.append(contactDuration)
        contactStartIdxs.append(contactStartIdx)
        contactEndIdxs.append(contactEndIdx)
    return contactDurations, tuple(zip(contactStartIdxs, contactEndIdxs))


def findSoundTriggerIdxs(timeStamps, contactFlags, soundDurationSec=2.0):
    """Find the time stamp indices where playing a sound was triggered
    (if sounds were on). This takes into account that a sound cannot
    be repeated until the current instance is completed. (Though I think
    two different sounds could overlap in time). All sounds for the 2024
    bootcamp were approx 3 seconds in duration, so I will use that here.
    """
    soundTriggerIdxs = []
    if np.any(contactFlags):
        contactIdxs = np.flatnonzero(contactFlags)
        contactTimes = timeStamps[contactFlags]
        soundStartTime = contactTimes[0]
        # Store first trigger idx
        soundTriggerIdxs.append(contactIdxs[0])
        while np.any(contactTimes > soundStartTime + soundDurationSec):
            soundStartTime = np.min(
                contactTimes[contactTimes > soundStartTime + soundDurationSec]
            )
            triggerIdx = contactIdxs[contactTimes == soundStartTime][0]
            soundTriggerIdxs.append(triggerIdx)
    return soundTriggerIdxs


from qt import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    Qt,
    QColor,
    QFont,
    QHeaderView,
    QSpacerItem,
    QSizePolicy,
    QLabel,
    QPixmap,
    QTextEdit,
)


def makeScoreQTable(
    entryList: List[LeaderboardEntry], parent: Optional[QDialog] = None
):
    """Make QTableWidget showing score data with one row per scopeRun
    in the list
    """
    table = QTableWidget(len(entryList), 6, parent)
    table.setHorizontalHeaderLabels(
        [
            "User",
            "Score",
            "Adv Time (s)",
            "# Contacts",
            "Contact Time (s)",
            "Wdr Time (s)",
        ]
    )
    table.setAlternatingRowColors(True)

    # style header
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.Stretch)
    header.setSectionResizeMode(1, QHeaderView.Stretch)
    header.setStyleSheet(
        """
      QHeaderView::section {
        background-color: lightgray;
        font-weight: bold;
      }
    """
    )

    # fill in rows
    for row, entry in enumerate(entryList):
        # username cell
        u = QTableWidgetItem(entry.userName)
        # score cell
        scoreStr = f"{int(entry.score)} {'*' if entry.flawlessFlag else ''}"
        s = QTableWidgetItem(scoreStr)
        # advTime cell
        advTimeW = QTableWidgetItem(f"{entry.scoreComponents['advancingTime'][0]:0.1f}")
        # number of contacts cell
        nContacts = entry.scoreComponents["contactCountPenalty"][0]
        nContactsW = QTableWidgetItem(f"{nContacts}")
        # contact time cell
        contactTime = entry.scoreComponents["contactTimePenalty"][0]
        contactTimeW = QTableWidgetItem(f"{contactTime:0.1f}")
        # withdrawal time cell
        wdrTime = entry.scoreComponents["withdrawalPenalty"][0]
        maxWithdrawalTime = 5
        minWithdrawalTime = 2
        if wdrTime > maxWithdrawalTime:
            extraTxt = " (too slow!)"
        elif wdrTime < minWithdrawalTime:
            extraTxt = " (too fast!)"
        else:
            extraTxt = ""
        wdrStr = f"{wdrTime:0.1f}{extraTxt}"
        wdrW = QTableWidgetItem(wdrStr)
        # Align cell widgets
        cellList = [u, s, advTimeW, nContactsW, contactTimeW, wdrW]
        for w in cellList:
            w.setTextAlignment(Qt.AlignCenter)
        # highlight the top entry
        if row == 0:
            highlight = QColor(255, 235, 205)  # very light peach
            for item in cellList:
                item.setBackground(highlight)
                f = QFont()
                f.setBold(True)
                item.setFont(f)
        for colIdx, cell in enumerate(cellList):
            table.setItem(row, colIdx, cell)
    # ensure rows fit content
    table.resizeRowsToContents()
    table.resizeColumnsToContents()

    # Explicitly set table height so that scroll bar doesn't appear
    tableHeight = (
        table.horizontalHeader().height
        + sum(table.rowHeight(r) for r in range(table.rowCount))
        + 2 * table.frameWidth
    )

    table.setMinimumHeight(tableHeight)
    table.setMinimumWidth(table.width)
    return table


def show_leaderboard(srList: List[ScopeRun], currentSr: Optional[ScopeRun] = None):
    """
    Pop up a “Leaderboard” dialog with centered text, alternating stripes,
    a styled header, and the top row highlighted + bolded.

    :param user_list: ordered list of objects, each with .username and .score
    :return: the QDialog instance (keep a reference so it isn't GC'd)
    """
    parent = slicer.util.mainWindow()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Leaderboard")

    # main layout
    layout = QVBoxLayout(dlg)

    if currentSr is not None and currentSr.valid:
        # Add a section for the Current Run
        # Title
        curLabel = QLabel("Current Trial", dlg)
        curLabel.setAlignment(Qt.AlignCenter)
        curLabel.setStyleSheet(
            "font-size: 12pt; font-weight: bold; font-style: italic; margin-bottom: 5px; margin-top: 5px"
        )
        # Report text
        reportText = currentSr.generateAnalysisReportText()
        textBox = QTextEdit(dlg)
        textBox.setReadOnly(True)  # don't allow editing/interaction
        textBox.setPlainText(reportText)
        textBox.document.setDocumentMargin(15)
        textBox.document.adjustSize()  # to trigger update
        documentHeight = textBox.document.size.height()
        marginHeight = (
            textBox.contentsMargins().top() + textBox.contentsMargins().bottom()
        )
        textBox.setFixedHeight(int(documentHeight + marginHeight))

        # Plot image
        newImPath = pathlib.Path(slicer.app.temporaryPath, "TempFancyPlot.png")
        fancyPlot6(currentSr, newImPath, "Current Trial Progress Plot")
        plotImg = createImageWidget(newImPath, dlg)
        # Table (one row for current run)
        curTable = makeScoreQTable([currentSr], dlg)
        layout.addWidget(curLabel)
        layout.addWidget(textBox)
        layout.addWidget(plotImg)
        layout.addWidget(curTable)
        # Calculate height
        curRunHeight = (
            curTable.minimumHeight
            + curLabel.height
            + plotImg.pixmap.height()
            + textBox.height
        )

    else:
        curRunHeight = 0

    # Title
    label = QLabel("Current Leaderboard", dlg)
    label.setAlignment(Qt.AlignCenter)
    label.setStyleSheet(
        "font-size: 12pt; font-weight: bold; font-style: italic; margin-bottom: 5px; margin-top: 5px"
    )
    # layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))
    layout.addWidget(label)
    titleHeight = label.height

    # table
    table = makeScoreQTable(srList, dlg)

    # compute a “just-big-enough” size
    total_w = (
        table.verticalHeader().width
        + sum(table.columnWidth(i) for i in range(table.columnCount))
        + 2 * table.frameWidth
    )
    total_h = table.minimumHeight + titleHeight + curRunHeight
    # add a little padding
    extraWidth = 60
    extraHeight = 30
    dlg.resize(total_w + extraWidth, total_h + extraHeight)

    layout.addWidget(table)
    # layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

    dlg.setLayout(layout)
    dlg.show()

    return dlg


from qt import QDialog, QVBoxLayout, QLabel, QPixmap


def createImageWidget(image_path, parent: Optional[QDialog] = None):
    """Create a QLabel widget with a pixmap of the input image path"""
    labelWidget = QLabel(parent)
    pixmap = QPixmap(image_path)

    if pixmap.isNull():
        labelWidget.setText("Failed to load image.")
    else:
        labelWidget.setPixmap(pixmap)
        labelWidget.setScaledContents(True)  # Optional: scales image to fit label
        # allow shrinkage:
        # labelWidget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    return labelWidget


def show_image_window(image_path):
    """
    Display an image in a Qt dialog window.

    :param image_path: full path to the image file
    :return: the QDialog instance
    """
    parent = slicer.util.mainWindow()
    dlg = QDialog(parent)
    dlg.setWindowTitle("Image Viewer")

    layout = QVBoxLayout(dlg)

    # Create label and set pixmap
    label = QLabel()
    pixmap = QPixmap(image_path)

    if pixmap.isNull():
        label.setText("Failed to load image.")
    else:
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # Optional: scales image to fit label
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    layout.addWidget(label)
    dlg.setLayout(layout)
    dlg.resize(pixmap.width(), pixmap.height())  # auto-size to image
    dlg.show()

    return dlg
