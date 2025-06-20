from pathlib import Path
import os

# run HelperClasses.py in interactor before running this script
""" 
recPath = r"C:/Users/mikeb/PlusApp-2.8.0.20191105-Win32/data/AirwayTrackerRec-MikeLeaderBoardTest1-2025-06-13-103016.mhd"
recDir = Path(r"C:/Users/mikeb/PlusApp-2.8.0.20191105-Win32/data/")
ecg3Path = Path(recDir, r"AirwayTrackerRec-ECG-2024-07-10-105153.mhd")
sess = Session({"userName": "TestUser1"})
rec = Recording(sess, ecg3Path)
sess.addRecording(rec)

gi = slicer.modules.ExampleGuideletWidget.guideletInstance
prog = gi.progressObj
zt = gi.zoneTuples
segNode = gi.parameterNode.GetNodeReference("airwayZoneSegmentationNode")
leafTform = gi.parameterNode.GetNodeReference("sceneLeafTransformNode")

rec.processRecordingToScopeRuns(leafTform, prog, zt, segNode)

zoneNames = list(rec.listOfScopeRuns[0].advPhase.zoneAnalysisDict.keys())
zoneNames = list([t[0] for t in zt])

for sr in rec.listOfScopeRuns:
    reportText = sr.generateAnalysisReportText()
    print(reportText)
    print("\n\n")
"""
# Make text + plot report
######################
gi = slicer.modules.ExampleGuideletWidget.guideletInstance
prog = gi.progressObj
zt = gi.zoneTuples
sessDir = r"C:\Users\mikeb\Documents\DynamicAirway\2024BootCamp\Analysis\BootcampSessionsOnly"  # now includes LAR3 file
listOfScopeRuns = loadAllScopeRunsFromDirectory(sessDir)
scoreInfo = []
for sr in listOfScopeRuns:
    sr.analyze(prog, zt)
    score, components = sr.calcScore()
    if score is not None:
        flawlessFlag = components["flawlessFlag"]
        scoreInfo.append((sr.userName, score, flawlessFlag))
# Sort
orderedScoreInfo = sorted(scoreInfo, key=lambda x: x[1])
for s in orderedScoreInfo:
    print(f"{s[0]:>6}: {int(s[1]): >5}{'*' if s[2] else ''}")

validScopeRuns = list([sr for sr in listOfScopeRuns if sr.valid])
# L = Leaderboard(validScopeRuns)
# L.display()


# Test Qt pixmap display
# imgPath = r"C:\Users\mikeb\OneDrive - SCH\Airway4D\Temp\TrackerFiles\2024\ToShare\Analysis\Images\SingleRunProgressPlots\TL_trial3_run11_progressFig.png"
# imDlg = show_image_window(imgPath)

newImPath = Path(slicer.app.temporaryPath, "TempFancyPlot.png")
sr = validScopeRuns[-2]
# fancyPlot6(sr, newImPath, sr.userName)
# mDlg = show_image_window(newImPath)

L = Leaderboard(validScopeRuns)
L.display(currentSr=sr)
