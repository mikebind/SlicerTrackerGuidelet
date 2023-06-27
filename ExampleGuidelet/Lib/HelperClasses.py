import logging
import os
import time

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



class Recording(object):
    def __init__(self, parentSessionObject, recordingFilePath, listOfScopeRuns=[]):
        logging.debug('Recording object init()')
        self.parentSession = parentSessionObject
        self.recordingFilePath = recordingFilePath
        self.listOfScopeRuns = listOfScopeRuns

    def processRecordingToScopeRuns(self):
        logging.debug('Recording.processRecordingToScopeRuns')


class ScopeRun(object):
    def __init__(self, parentRecordingObject, timestamps, positions, orientations):
        logging.debug('ScopeRun object init()')
        self.parentRecording = parentRecordingObject
        self.timeStamps = timestamps
        self.positions = positions
        self.orientations = orientations

    def saveToFile(self, saveDir):
        logging.debug('ScopeRun.saveToFile()')


