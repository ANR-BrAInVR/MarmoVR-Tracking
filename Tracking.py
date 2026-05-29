# -*- coding: utf-8 -*-
"""
Created on Mon May 11 12:30:45 2022

@authors: Manuel

TODO:
- TCP server multi-client...
- sync cameras, use xiApi software triggering mode (prevent differential motion) :
   https://www.ximea.com/support/wiki/apis/XiAPI_Camera_Trigger_and_Synchronization_Signals

"""

import multiprocessing as mp
import threading
import socket
import time
import sys
import os
import shutil
import re
import csv
from ctypes import c_wchar_p
import numpy as np
from ximea import xiapi
import cv2
from PyQt5.QtWidgets import *

# IPs and communication ports (warning: ports indexed by camNb not camInd)
IP = {'Tracking': '192.168.0.2', 'Rendering': '192.168.0.1', 'Recobox0': '192.168.0.20', 'Recobox1': '192.168.0.21'}
UDPserverRendering = (IP['Rendering'], 50771)
TCPserverTracking = (IP['Tracking'], 65432)
TCPserverRecobox = [(IP['Recobox0'], 2390), (IP['Recobox1'], 2391)]
UDPpacketSize = 65000   # Size of UDP packets
TCPpacketSize = 4096    # Size of TCP packets

# Image global settings
imgWidth, imgHeight = 1280, 1024

# Inferred keypoint positions by DeepLabCut (DLC)
keyColors = {'eartuffL': (0, 0, 255), 'eartuffR': (0, 255, 0), 'whitepatch': (255, 255, 255), 'snout': (200, 200, 200), 'backneck': (200, 200, 200),
             'shoulderL': (0, 0, 145), 'shoulderR': (0, 145, 0), 'wristL': (0, 0, 90), 'wristR': (0, 90, 0), 'hipL': (0, 0, 145), 'hipR': (0, 145, 0),
             'ankleL': (0, 0, 90), 'ankleR': (0, 90, 0), 'tailroot': (200, 200, 200), 'tailmiddle': (145, 145, 145), 'tailtip': (95, 95, 95)}
nKeys = len(keyColors)
keyRadius = 3  # Maximal radius size of the keypoints for monitoring (reached when inference p=1)
cyclopRadius = 4  # Maximal radius size of cyclop for monitoring (reached when inference p=1)
cyclopColor = 0  # Detection or cyclop center color (white)

# Settings place holder
class Settings:
    owner = None

# Tracking that detects 2D position of animals for each camera, triangulates and runs DLC on cropped images
class Tracking:

    # Constructor with param initialization
    def __init__(self):
        """Initializes experiment's parameters"""

        # Log object
        self.log = Log(logLevel=2, showTime=True)
        self.log.LogText(1, 'Tracking() called')

        # Load settings and set important parameters
        self.LoadSettings()
        self.camCount = len(self.camList)       # Number of cameras in the setup

        # Global control flags (communication between processes)
        self.startRequest = mp.Value('B', False)             # Start video acquisition when True
        self.expStarted = mp.Value('B', False)               # Experiment started when True (acquiring images and monitoring)
        self.trialStarted = mp.Value('B', False)             # Trial started when True
        self.stopRequest = mp.Value('B', False)              # stop video acquisition when True
        self.quit = mp.Value('B', False)                     # Quit program when True (stop acquisition)
        self.dataRecording = mp.Value('B', False)            # Start/stop recording data when True/False
        self.serverRunning = mp.Value('B', False)            # TCP server on/off when True/False (used to launch the GUI)
        self.recoboxRunning = mp.Value('B', False)           # TCP server of Recobox on/off when True/False
        self.acquiring = mp.Array('B', [False] * self.camCount)        # Acquiring when True (for each camera)
        self.DLCrunning = mp.Value('B', False)                       # DLC inferences running on all cameras
        self.syncRequest = mp.Array('B', [False] * self.camCount)      # Synch request (reset startIndexes) when True (for each camera)
        self.newRefRequest = mp.Array('B', [False] * self.camCount)    # Request new reference images when True (for each camera)
        self.videoReady = mp.Array('B', [False] * self.camCount)       # Stores the video writers to add sound channel in the end if desired

        # Manager to share lists of objects across processes and threads
        manager = mp.Manager()

        # Events
        self.eventString = manager.Value(c_wchar_p, '')              # String containing the last event to be stored with results

        # Images processed Ximea stream
        self.imgMonits = manager.list([None] * self.camCount)          # Monitoring images, one per camera (for resized images: full, diff, thresh, morph)
        self.imgCrops = manager.list([None] * self.camCount)           # Cropped images, one per camera (not resized, used for DLC or monitoring)
        self.imgCropULs = manager.list([[0, 0]] * self.camCount)       # Upper-left position (X,Y) of cropped image in camera coordinates (full image)
        self.imgIndexes = mp.Array('l', [-1] * self.camCount)          # Index of last image received in each camera (stream)

        # Detected 2D position of animal in each camera
        self.pos2Ds = manager.list([np.zeros(2)] * self.camCount)      # Positions 2D of mass center in camera coordinates (full image)

        # Triangulated 3D position of animal
        self.pos3D = manager.list([np.zeros(3)])                 # Position 3D mass center in cave coordinates (valid pos2D triangulated)
        self.imgIndexPos3D = mp.Value('l', -1)                   # Index of cam0 image from which pos2D have been triangulated (stream)

        self.pos2D_cyclop = manager.list([np.zeros(3)] * self.camCount)            # Computed cyclop 2D positions for each camera (needs DLC activated, in full image coordinates)
        self.pos3D_cyclop = manager.list([np.zeros(4)])                          # Triangulated cyclop 3D positions (X, Y, Z, pMean) (in aquarium coordinates)

        self.pos2Ds_DLC = manager.list([np.zeros((nKeys, 3))] * self.camCount)      # 2D positions and confidence (X, Y, p) of each inferred keypoint (in full image coordinates)
        self.imgIndexes_DLC = mp.Array('l', [-1] * self.camCount)                  # Index of image from which keypoints have been inferred

        # Triangulated DLC inferred keypoints
        self.pos3Ds_DLC = manager.list([np.zeros((nKeys, 4))])        # Triangulated 3D positions (X, Y, Z, pMean) of each inferred pair (in aquarium coordinates)
        self.imgIndexPos3D_DLC = mp.Value('l', -1)                   # Index of cam0 image from which DLC keypoints have been inferred

        # Trial infos for results directory and filename of results, and to update UI
        self.expID = manager.Value(c_wchar_p, 'exp')
        self.subjectID = manager.Value(c_wchar_p, 'subj')
        self.trialID = manager.Value(c_wchar_p, 'trial')
        self.condID = manager.Value(c_wchar_p, 'cond')

        # UI shared controller vars
        self.runDetect = mp.Value('B', self.settings.runDetect)
        self.runDLC = mp.Value('B', self.settings.runDLC)
        self.triangulate = mp.Value('B', self.settings.triangulate)
        self.showPos2D = mp.Value('B', self.settings.showPos2D)
        self.showDLC = mp.Value('B', self.settings.showDLC)
        self.useCyclop = mp.Value('B', self.settings.useCyclop)
        self.sendPos3D = mp.Value('B', self.settings.sendPos3D)
        self.saveResults = mp.Value('B', self.settings.saveResults)
        self.imgMode = manager.list(self.settings.imgMode)                                  # Image monitoring mode
        self.recSound = mp.Value('B', self.settings.recSound)        # Records sound during trials
        self.imgMode = manager.Value(c_wchar_p, self.settings.imgMode)                  # Image monitoring mode ('full', 'crop', 'diff', 'thresh', 'morph', 'none')

        # Run sanity check for newly loaded settings
        self.CheckSettings(atLoad=True)

        # Load sound libraries if record requested
        if self.recSound.value:
            import pyaudio  # Audio devices
            import wave  # Handles wav data
            from pydub import AudioSegment  # Convert .wav to .flac or .mp3
            from moviepy import VideoFileClip, AudioFileClip  # Used to add audio channel to video

        # Initialize crop images
        self.imgCropDim = (self.cropSize[1], self.cropSize[0], 3)
        self.imgCrops[:self.camCount] = [np.zeros(self.imgCropDim, 'uint8')] * self.camCount
        self.imgIndexes[:self.camCount] = [0] * self.camCount

        # Global control flags
        self.serverRunning.value = False
        self.expStarted.value = False
        self.trialStarted.value = False
        self.startRequest.value = False
        self.stopRequest.value = False
        self.quit.value = False
        self.acquiring[:self.camCount] = [False] * self.camCount
        self.syncRequest[:self.camCount] = [False] * self.camCount
        self.newRefRequest[:self.camCount] = [False] * self.camCount
        self.videoReady[:self.camCount] = [False] * self.camCount
        self.dataRecording.value = False

        # Check if Recobox connected
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:
            try:
                self.recoboxRunning.value = True
                clientSocket.connect(TCPserverRecobox[0])
            except socket.error:
                self.recoboxRunning.value = False
        self.log.LogText(2, 'Tracking: Recobox[0] is %s' % ('active' if self.recoboxRunning.value else 'inactive'))
        # self.recoboxRunning.value = False       # DEBUG: force recobox active (for UI)

        # Starts TCP server to receive commands from Rendering PC
        TCPserverThread = threading.Thread(target=self.TCPserver, args=())
        TCPserverThread.start()

        # Waits TCP server to be up and running
        while not self.serverRunning.value: pass
        time.sleep(0.1)

        # Starts GUI
        self.GUIproc = mp.Process(target=self.StartGUI)
        self.GUIproc.start()
        self.log.LogText(2, 'Tracking: GUI process started')

        while True:
            # self.log.LogText(2, 'Tracking: %d' % self.startRequest.value)

            if self.startRequest.value:
                self.startRequest.value = False
                if self.expStarted.value:
                    self.log.LogText(2, 'Tracking: start request received but ignored (already started)')
                else:
                    self.log.LogText(2, 'Tracking: start request received')

                    # Starts tracking
                    self.Start()

            if self.stopRequest.value:
                if not self.expStarted.value:
                    self.log.LogText(2, 'Tracking: stop request received but ignored (not started)')
                    time.sleep(0.2)
                    self.stopRequest.value = False
                else:
                    self.log.LogText(2, 'Tracking: stop request received')
                    time.sleep(0.2)

                    # Closes openCV windows if any
                    cv2.destroyAllWindows()

                    # Back to idle status (all unnecessary threads and processes are closed now)
                    self.acquiring[:self.camCount] = [False] * self.camCount
                    self.expStarted.value = False
                    self.trialStarted.value = False
                    self.stopRequest.value = False
                    self.startRequest.value = False
                    self.log.LogText(2, 'Tracking: waiting for start request')

            if self.quit.value:
                self.log.LogText(2, 'Tracking: quit request received')
                break

    def __del__(self):
        """Destructor"""

        self.log.LogText(1, 'Tracking destructor called')

        # Close openCV windows if any
        cv2.destroyAllWindows()

        # Quit GUI if still there
        # self.GUIproc.kill()

        # Give time for processes to end
        time.sleep(0.5)

    def LoadSettings(self):
        """Loads settings files"""

        self.log.LogText(1, 'LoadSettings() called')

        def LoadSettingsFile(self, settingsName, storeVariable='self'):
            """Loads specific settings to self or another object's attributes"""

            filename = os.path.join('Settings', settingsName + '.txt')
            if not os.path.isfile(filename):
                return

            with open(filename, 'r') as fSet:
                settingLines = fSet.readlines()
                for settingLine in settingLines:
                    settingArgs = re.split('\t+', settingLine)
                    if len(settingArgs) < 2 or settingArgs[0][0] == '#':
                        continue
                    exec('{}.{}={}'.format(storeVariable, *settingArgs))

            self.log.LogText(2, '\'%s\' loaded' % settingsName)

        # System settings place-holder
        self.settings = Settings()

        # Gets experiment settings
        LoadSettingsFile(self, 'System settings', storeVariable='self.settings')
        LoadSettingsFile(self, 'Camera settings')
        LoadSettingsFile(self, 'Detection settings')

        self.log.logLevel = self.settings.logLevel  # Updates log level

    def CheckSettings(self, atLoad=True):
        """Run sanity check on loaded settings"""

        self.log.LogText(1, 'CheckSettings() called')

        if atLoad:
            # Loads settings to UI shared controller vars
            self.runDetect.value = self.settings.runDetect
            self.runDLC.value = self.settings.runDLC
            self.triangulate.value = self.settings.triangulate
            self.showPos2D.value = self.settings.showPos2D
            self.useCyclop.value = self.settings.useCyclop
            self.showDLC.value = self.settings.showDLC
            self.sendPos3D.value = self.settings.sendPos3D
            self.imgMode.value = self.settings.imgMode
            self.saveResults.value = self.settings.saveResults

        # Update camCount, trial infos and system infos
        self.camCount = len(self.camList)

        # If blob detector, thresh or morph depend on blobDetectMode
        if not self.runDetect.value:
            if self.imgMode.value in ['diff', 'thresh', 'morph']:
                self.log.LogText(2, 'CheckSettings: cannot use imgMode=\'%s\' without runDetect=True, switching to \'full\'' % self.imgMode.value)
                self.imgMode.value = 'full'

        # If no detection
        if not self.runDetect.value and not self.runDLC.value:
            self.log.LogText(2, 'CheckSettings: no detection enabled (runDetect and runDLC are set to False)')
            self.triangulate.value = False
            self.showPos2D.value = False
            self.showDLC.value = False
            self.sendPos3D.value = False
            self.imgMode.value = 'full'

        # If no DLC
        if not self.runDLC.value:
            self.useCyclop.value = False
            self.getMotionDir = False
            self.getGazeDir = False

        if not self.getVelocity and not self.getGazeDir:
            self.getMotionDir = False

    def Start(self):
        """Starts video processing and tracking"""

        self.log.LogText(1, 'Start() called')

        processList = []

        # Start video processing to get detected 2D positions and cropped images for DLC
        videoProcessingProcs = []
        self.acquiring[:self.camCount] = [False] * self.camCount
        for camInd in range(self.camCount):
            videoProcessingProcs.append(mp.Process(target=self.VideoProcessing, args=(camInd,)))
        time.sleep(0.1)
        for camInd in range(self.camCount):
            videoProcessingProcs[camInd].start()
        processList.extend(videoProcessingProcs)

        # Waits that all video processing processes are acquiring from cameras
        t0 = time.perf_counter()
        while self.acquiring[:self.camCount] != [True] * self.camCount:
            time.sleep(0.1)
            if (time.perf_counter() - t0 > self.connectTimeout + 1) or self.stopRequest.value:
                self.log.LogText(2, 'Start: could not start all VideoProcessing processes')
                self.stopRequest.value = True
                return -1
        self.expStarted.value = True
        time.sleep(0.1)

        # Send sync request to reset image indexes
        self.log.LogText(2, 'Start: sending syncRequest')
        self.syncRequest[:self.camCount] = [True] * self.camCount
        while self.syncRequest[:self.camCount] != [False] * self.camCount:
            time.sleep(0.1)

        # Start DLC inferences process (which will start 4 threads)
        if self.runDLC.value:
            self.DLCrunning.value = False
            runDLCproc = mp.Process(target=self.RunDLC, args=())
            runDLCproc.start()
            processList.append(runDLCproc)
            while not self.DLCrunning.value:
                time.sleep(0.1)

        if self.runDetect.value or self.runDLC.value or self.recSound.value:
            # Start triangulation process (only one)
            if self.triangulate.value:
                self.log.LogText(1, 'Start: start triangulation')
                triangulationProc = mp.Process(target=self.Triangulation, args=())  # pCutoff not used (image thresholding only)
                triangulationProc.start()
                processList.append(triangulationProc)
            time.sleep(0.25)

            # Start result saving process, 2D pos (per camera) and triangulated 3D pos (detect and DLC)
            saveResultsProc = mp.Process(target=self.SaveResults, args=(self.framerate, ))
            saveResultsProc.start()
            processList.append(saveResultsProc)
            time.sleep(0.25)

        # Show 4 cropped images in one with DLC keypoints if requested
        self.Monitoring()  # Blocking (main thread)

        # Waits until all processes are dead to avoid broken pipes
        self.log.LogText(2, 'Start: waiting for processes to end')
        aliveCount = len(processList)
        while aliveCount != 0:
            aliveCount = 0
            for proc in processList:
                if proc.is_alive():
                    aliveCount += 1
        self.log.LogText(2, 'Start: processes ended')

        # End of acquisition, ends nicely
        cv2.destroyAllWindows()

    # Monitoring camera images and tracking results (detection and DLC)
    def Monitoring(self, showPerfs=True):
        """Monitoring of the experiment tracking (blocking in main thread)"""

        self.log.LogText(1, 'Monitoring() called as main thread')

        windowName = 'Images %s' % ('(with DLC)' if self.runDLC.value else '')
        cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        imgIndexesPrev = [-1] * self.camCount
        imgMode_stored = 'None'

        # Initializes performance profiler
        if showPerfs:
            tPerf = time.perf_counter()
            missed = 0
            perfStr = ''

        while True:

            # Quits on stopAcquisition
            if self.stopRequest.value:
                cv2.destroyAllWindows()
                self.log.LogText(1, 'Monitoring: stop acquisition received, quitting')
                return

            # Prepares target panel dimensions
            if imgMode_stored != self.imgMode.value:
                imgMode_stored = self.imgMode.value
                resized = not (self.imgMode.value == 'crop')
                if resized:
                    panelDim = [imgHeight, imgWidth]
                    imgMonitDim = [imgHeight // 2, imgWidth // 2]
                else:
                    panelDim = [self.cropSize[0] * 2, self.cropSize[1] * 2]
                    imgMonitDim = list(self.cropSize)
                self.log.LogText(2, 'Monitoring: changing to imgMode=\'%s\'' % self.imgMode.value)

                # Resetting panel images
                imgIndexesPrev = [-1] * self.camCount
                textColor = [255] * 3
                imgPanel = np.zeros(panelDim + [3], 'uint8')

            # Process new set of images
            sameCounter = 0
            for camInd in range(self.camCount):

                # Continues if same image
                if self.imgIndexes[camInd] == imgIndexesPrev[camInd]:
                    sameCounter += 1
                    # print('(cam%d) sameCounter=%d' % (camInd, sameCounter))
                    continue

                # Update performance counter
                if showPerfs:
                    missed += max(0, self.imgIndexes[camInd] - imgIndexesPrev[camInd] - 1)   # Missed frames
                    # print('(cam%d) missed=%d' % (camInd, missed))
                imgIndexesPrev[camInd] = self.imgIndexes[camInd]

                # Get last image in the desired format
                camNb = self.camList[camInd]
                self.log.LogText(4, 'Monitoring: getting camera %d image %d' % (camNb, self.imgIndexes[camInd]))
                cropULs = self.imgCropULs[camInd].copy()
                if resized:
                    img = np.copy(self.imgMonits[camInd])    # Get resized image (for full, diff, thresh or morph)
                else:
                    img = np.copy(self.imgCrops[camInd])     # Get cropped image
                if img.shape[0] != imgMonitDim[0]:
                    self.log.LogText(3, 'Monitoring: image size not matching, %s != %s' % (img.shape, tuple(imgMonitDim)))
                    continue

                # Show crop rectangle
                if resized:
                    img = cv2.rectangle(img, np.array(cropULs)//2, (cropULs + np.array(self.cropSize))//2, textColor, thickness=1)

                if self.showDLC.value:
                    # Adds keypoints inferred by DLC (with lag when shown in real-time)
                    posKey = np.copy(self.pos2Ds_DLC[camInd])  # Get tensor flow detected pos2D
                    for keyInd, keyName in enumerate(self.keyNames):
                        # Draws circle on inferred keypoint (size depends on probability)
                        if resized:
                            xKey, yKey = (posKey[keyInd, :2]/2).astype(int)
                        else:
                            xKey, yKey = posKey[keyInd, :2].astype(int) - cropULs
                        rKey = int(keyRadius * np.sqrt(posKey[keyInd, 2]))     # Circles with radius depending on inference confidence
                        # rKey = 5    # DEBUG
                        self.log.LogText(4, 'Monitoring: on cam%d, keypoint [%s] is drawn at (%d, %d) with r=%d' % (camNb, keyName, xKey, yKey, rKey))
                        img = cv2.circle(img, (xKey, yKey), rKey, keyColors[keyName], thickness=cv2.FILLED)

                if self.showPos2D.value:
                    if self.runDLC.value and self.useCyclop.value:
                        # Draws target on cyclop (if valid)
                        posCyclop = np.copy(self.pos2D_cyclop[camInd])       # Get copy of cyclop position
                        if posCyclop[2] != -1:
                            rPos = np.sqrt(posCyclop[2])      # Inference probability
                            # rPos = 5    # DEBUG
                            if resized:
                                xPos, yPos = (posCyclop[:2]/2).astype(int)
                            else:
                                xPos, yPos = posCyclop[:2].astype(int) - cropULs
                            img = cv2.circle(img, (xPos, yPos), int(cyclopRadius * rPos), [255-cyclopColor]*3, thickness=cv2.FILLED)
                            img = cv2.circle(img, (xPos, yPos), int(keyRadius * rPos), [cyclopColor]*3, thickness=cv2.FILLED)

                    elif self.runDetect.value:
                        # Draws target on detected 2D position
                        pos2D = np.copy(self.pos2Ds[camInd])       # Get copy of pos2D positions
                        if resized:
                            xPos, yPos = (pos2D/2).astype(int)
                        else:
                            xPos, yPos = pos2D.astype(int) - cropULs
                        img = cv2.circle(img, (xPos, yPos), int(cyclopRadius), [255-cyclopColor]*3, thickness=cv2.FILLED)
                        img = cv2.circle(img, (xPos, yPos), int(keyRadius), [cyclopColor]*3, thickness=cv2.FILLED)

                # # Resize images when full
                # if resized:
                #     img = cv2.resize(img, (imgWidth // 2, imgHeight // 2))

                # Rotate image to align monitoring with setup
                if not resized:
                    if self.rotateCamList[camNb] == 90:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif self.rotateCamList[camNb] == -90:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    elif self.rotateCamList[camNb] == 180:
                        img = cv2.rotate(img, cv2.ROTATE_180)

                # Add camera label
                img = cv2.putText(img, 'Cam%d' % camNb, (10 if camInd in [0, 2] else imgMonitDim[1]-55, imgMonitDim[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor)

                # Places updated image in 4-images panel (for monitoring)
                (i, j) = [(1, 0), (1, 1), (0, 0), (0, 1)][camNb]
                imgPanel[i * imgMonitDim[0]:(i + 1) * imgMonitDim[0], j * imgMonitDim[1]:(j + 1) * imgMonitDim[1], :] = img

            # If none of the images changed, continues
            if sameCounter == self.camCount:
                continue

            # Compute framerate
            if showPerfs:
                t1 = time.perf_counter()
                if t1 - tPerf > 1:
                    # Every one second, estimates missed frames
                    missedFPS = self.framerate - missed / self.camCount
                    perfStr = 'Performance: %.f/%d frames (%.1f%%)' % (missedFPS, self.framerate, 100 * missedFPS / self.framerate)
                    missed = 0
                    tPerf = t1
                imgPanel = cv2.putText(imgPanel, perfStr, (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor)
            imgPanel = cv2.putText(imgPanel, 'Mode: \'%s\'' % self.imgMode.value, (imgMonitDim[1]-40, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor)

            # Add retained triangulation value
            if self.triangulate.value:
                tpos3D = tuple(self.pos3D_cyclop[0][:3]) if self.useCyclop.value else tuple(self.pos3D[0])
                imgPanel = cv2.putText(imgPanel, 'Triangulation (%.1f, %.1f, %.1f)' % tpos3D, (imgMonitDim[1]-100, panelDim[0]-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=textColor)

            # Shows image panel
            cv2.imshow(windowName, imgPanel)
            cv2.moveWindow(windowName, 350, 0)      # TODO: move window but allow user to move it
            cv2.waitKey(1)

    # Ximea camera video processing (PROCESS)
    def VideoProcessing(self, camInd, showPerfs=False):
        """For each camera, detects 2D position, crops images around animal (with UL corner)"""

        camNb = self.camList[camInd]
        self.log.LogText(1, 'VideoProcessing(%d) process running (PID: %d)' % (camNb, mp.current_process().pid))

        nframePrev = -1             # Frame number returned by XiAPI
        indexStart = -1             # To 'softly' synchronize image indexes
        dataRecordingPrev = False   # Stores previous flag state
        video = None                # Stores video writer
        self.acquiring[camInd] = False

        # Default detected 2D position of animal (image center)
        pos2D = [imgWidth//2, imgHeight//2]

        # Get camera from Ximea API
        xiCam = xiapi.Camera()
        self.log.LogText(2, 'VideoProcessing(%d): Opening camera SN=%s' % (camNb, self.camSerialNbs[camNb]))

        t0 = time.perf_counter()
        while True:
            try:
                xiCam.open_device_by_SN(self.camSerialNbs[camNb])
                break
            except Exception as errMsg:
                self.log.LogText(2, 'VideoProcessing(%d): could not open camera (%s), trying later' % (camNb, errMsg))
            if (time.perf_counter() - t0 > self.connectTimeout) or self.stopRequest.value:
                self.log.LogText(1, 'VideoProcessing(%d): could not open camera, quitting' % camNb)
                self.stopRequest.value = True
                return -1

        # Set camera parameters
        xiCam.set_param('recent_frame', 0)
        xiCam.set_acq_timing_mode('XI_ACQ_TIMING_MODE_FRAME_RATE')
        xiCam.set_param('framerate', self.framerate * self.frameStep)
        xiCam.set_exposure(self.exposure)
        xiCam.set_gain(self.gain)
        xiCam.set_imgdataformat('XI_RGB24')
        self.log.LogText(2, 'VideoProcessing(%d): xiCam params loaded' % camNb)

        # Reference image Average placeholder
        imgRefAvg = np.zeros((imgHeight, imgWidth, 3))
        imgRefCount = 0

        # Load mask image used to remove cave background
        if self.useExclMask:
            filename = 'Masks/maskCam%d.png' % camNb
            imgMask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.log.LogText(2, 'VideoProcessing(%d): \'%s\' mask image loaded' % (camNb, filename))

        # 2D detection inits
        if self.runDetect.value:

            # Load reference image of this camera
            filename = 'References/refCam%d.png' % camNb
            imgRefGray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            self.log.LogText(2, 'VideoProcessing(%d): \'%s\' reference image loaded' % (camNb, filename))

            # Get threshold value according to camera position (upper and lower different light intensity)
            if camNb < 2:
                camThresh = self.lowerCamThresh     # 0 and 1 are lower cameras
            else:
                camThresh = self.upperCamThresh     # 2 and 3 are upper cameras
            self.log.LogText(2, 'VideoProcessing(%d): threshold value set' % camNb)

            # Initialize morphing kernel and dynamic mask
            kernel = np.ones((int(self.morphMaskSize[0]), int(self.morphMaskSize[1])), np.uint8)
            if self.useDynMask:
                dynMask = np.ones((imgHeight, imgWidth), 'uint8') * 255
                xDMS, yDMS = self.dynamicMasksize

            # Initialize pos2D filter (average over sliding window)
            posXslide = [imgWidth//2] * self.filter2Dsize
            posYslide = [imgHeight//2] * self.filter2Dsize

            self.log.LogText(2, 'VideoProcessing(%d): dynamic mask and 2D filter initialized' % camNb)

        # Create instance of Image to store image data and metadata
        xiImg = xiapi.Image()

        # Start data acquisition
        xiCam.start_acquisition()
        self.log.LogText(2, 'VideoProcessing(%d): xiCam data acquisition started' % camNb)

        # Profiler and perfs inits
        showProfiler = camNb in self.profilerList       # Time debug
        if showProfiler or showPerfs:
            tm = t0 = time.perf_counter()
            missed = 0
            missedCount = 0

        self.acquiring[camInd] = True
        while True:

            if showProfiler:       # Timing debug
                T = []
                Tind = 0
                T += [time.perf_counter()]
                print('  T(%d)=%f ms' % (Tind, T[Tind]*1000))

            # Quits on stopRequest
            if self.stopRequest.value or self.quit.value:
                self.log.LogText(1, 'VideoProcessing(%d): %s requested' % (camNb, 'stop acquisition' if self.stopRequest.value else 'quit'))
                self.acquiring[camInd] = False
                self.dataRecording.value = False     # Sends signal to stop data recording (in case)
                if not dataRecordingPrev:
                    # Quits when no longer recording
                    cv2.destroyAllWindows()
                    if xiCam is not None:       # Closes camera if opened
                        xiCam.stop_acquisition()
                        xiCam.close_device()
                        self.log.LogText(2, 'VideoProcessing(%d): camera closed, quitting' % camNb)
                    return

            # At trial start: create video writers for each camera
            if self.dataRecording.value and not dataRecordingPrev:
                # Creates paths if not existing
                path = self.resultsDir + '%s/%s/' % (self.expID.value, self.subjectID.value)
                resultsFile = 'Trial-%s_Cond-%s' % (self.trialID.value, self.condID.value)
                try:
                    os.makedirs(path)
                except FileExistsError:
                    pass
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(path + resultsFile + '_cam%d.mp4' % camNb, fourcc, self.framerate, (imgWidth, imgHeight), isColor=True)
                self.log.LogText(2, 'VideoProcessing(%d): video writers created' % camNb)
                dataRecordingPrev = True

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - inits' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000))

            # Get data and pass them from camera to img
            xiCam.get_image(xiImg)

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - xiCam.get_image() (%03d)' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000, xiImg.nframe - indexStart))

            # Skip frames as requested by frameStep
            if self.frameStep > 1 and (xiImg.nframe - nframePrev) % self.frameStep != 0:
                continue

            # Store newly acquired image index
            if self.syncRequest[camInd] or indexStart == -1:
                indexStart = xiImg.nframe
                self.syncRequest[camInd] = False
            imgIndex = (xiImg.nframe - indexStart) // self.frameStep

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - sync and indexing, image %03d (%03d)' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000, imgIndex, xiImg.nframe - indexStart))

            # Stores newly acquired image numpy array
            img = xiImg.get_image_data_numpy()
            self.log.LogText(3, 'VideoProcessing(%d): new image acquired %03d' % (camNb, imgIndex))       # Timing debug

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - xiImg.get_image_data_numpy()' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000))

            # During trial: update corresponding video
            if self.dataRecording.value and dataRecordingPrev:
                video.write(img)

            # At trial end, closes video writers and saves
            elif dataRecordingPrev:                
                video.release()
                video = None
                self.log.LogText(2, 'VideoProcessing(%d): video closed and saved' % camNb)
                self.videoReady[camInd] = True
                dataRecordingPrev = False

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - video.write()' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000))

            # Process new reference request
            if self.newRefRequest[camInd]:
                if imgRefCount == 0:
                    # Start
                    self.log.LogText(2, 'VideoProcessing(%d): starting image sampling for new reference image' % camNb)
                    frameCount = 0
                if frameCount % self.refFrameSample == 0:
                    # Add image to average
                    imgRefAvg = imgRefAvg + img
                    imgRefCount += 1
                frameCount += 1
                if imgRefCount == self.refNframes:
                    # Sampling finished, computes average
                    imgRefAvg = imgRefAvg / self.refNframes
                    imgRefAvg = imgRefAvg.astype('uint8')
                    # Converts to grayscale (tracking is off so not done before) and update file for this camera
                    imgRefGray = cv2.cvtColor(imgRefAvg, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite('References/refCam%d.png' % camNb, imgRefGray)
                    self.log.LogText(2, 'VideoProcessing(%d): new reference image updated and stored (%d images averaged)' % (camNb, self.refNframes))
                    # Resets variables
                    imgRefAvg = np.zeros((imgHeight, imgWidth, 3))
                    imgRefCount = 0
                    self.newRefRequest[camInd] = False

            # Detects position of animal in image
            if self.runDetect.value:

                processFrame = True         # Needs running simple detection on this frame ?

                if self.dynCrop and self.runDLC:
                    # Dynamic crop based on DLC inferred keypoints, if valid keypoints exist
                    keysDetected = self.pos2Ds_DLC[camInd][:, 2] > self.detectThresh
                    if keysDetected.sum() >= self.nKeysDynCrop:       # was if np.any(keysDetected):
                        # Get center of mass of detected positions
                        posX = np.average(self.pos2Ds_DLC[camInd][keysDetected, 0])
                        posY = np.average(self.pos2Ds_DLC[camInd][keysDetected, 1])
                        processFrame = False
                        self.log.LogText(4, 'VideoProcessing(%d): image %03d using keypoint\'s center of mass (X,Y)=(%.1f, %.1f)' % (camNb, imgIndex, posX, posY))

                if showProfiler:  # Timing debug
                    Tind += 1
                    T += [time.perf_counter()]
                    print('    T(%d)-T(%d)=%f ms - runDetect start' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                if processFrame:
                    # Prepares grayscale image
                    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    if showProfiler:  # Timing debug
                        Tind += 1
                        T += [time.perf_counter()]
                        print('      T(%d)-T(%d)=%f ms - cvtColor()' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                    # Subtracts the reference image (grayscale)
                    if self.subtractMethod == 0:        # absolute diff (all cases)
                        imgDiff = np.abs(imgGray.astype('int16') - imgRefGray.astype('int16')).astype('uint8')
                    elif self.subtractMethod == 1:  # ref-img (darker animal)
                        imgDiff = cv2.subtract(imgRefGray, imgGray)
                    elif self.subtractMethod == 2:  # img-ref (lighter animal)
                        imgDiff = cv2.subtract(imgGray, imgRefGray)

                    if showProfiler:  # Timing debug
                        Tind += 1
                        T += [time.perf_counter()]
                        print('      T(%d)-T(%d)=%f ms - subtract()' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                    # Threshold the diff image (binary)
                    _, imgThresh = cv2.threshold(imgDiff, camThresh, 255, cv2.THRESH_BINARY)

                    if showProfiler:  # Timing debug
                        Tind += 1
                        T += [time.perf_counter()]
                        print('      T(%d)-T(%d)=%f ms - threshold()' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                    # Bitwise operation with exclusion mask
                    if self.useExclMask:
                        imgThresh = cv2.bitwise_and(imgThresh, imgMask)

                    # Bitwise operation with dynamic mask
                    if self.useDynMask:
                        imgThresh = cv2.bitwise_and(imgThresh, dynMask)

                    # Morphing
                    imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_OPEN, kernel)

                    if showProfiler:  # Timing debug
                        Tind += 1
                        T += [time.perf_counter()]
                        print('      T(%d)-T(%d)=%f ms - morphologyEx()' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                    # Compute moments
                    M = cv2.moments(imgMorph)
                    #     TODO: 1. if computing values not needed try other fuctions
                    #           2. check if CUDA is used (faster), if not use cuda optimized cv functions
                    #           3. use a threshold for valid detection so that when no animal detected returns pos2D=[-1,-1]

                    if M['m00'] != 0:   # TODO: use small criteria (before it was rounded to an int)
                        # Something detected: use moments for new position
                        posX = M['m10'] / M['m00']
                        posY = M['m01'] / M['m00']

                        if self.useDynMask:
                            # Update dynamic mask to new position
                            dynMask = np.zeros((imgHeight, imgWidth), 'uint8')
                            dynMask[max(int(posY)-yDMS, 0):max(int(posY)+yDMS, imgHeight), max(int(posX)-xDMS, 0):max(int(posX)+xDMS, imgWidth)] = 255
                    else:
                        # Nothing detected (animal moved out of dynamic mask): keep previous position
                        # posX = posXslide[0]
                        # posY = posYslide[0]
                        # Nothing detected (animal moved out of dynamic mask): take center of image
                        # posX = imgWidth // 2
                        # posY = imgHeight // 2
                        # Nothing detected (animal moved out of dynamic mask): take -1, -1
                        posX = -1
                        posY = -1

                        if self.useDynMask:
                            # Will take full image for next detection
                            dynMask = np.ones((imgHeight, imgWidth), 'uint8') * 255

                    # 2D filtering: average over sliding window  to stabilize
                    if self.filter2D == 1:
                        posXslide = [posX] + posXslide[:-1]
                        posYslide = [posY] + posYslide[:-1]
                        posX = np.mean(posXslide)
                        posY = np.mean(posYslide)

                    self.log.LogText(3, 'VideoProcessing(%d): image %03d detected position (X,Y)=(%.1f, %.1f)' % (camNb, imgIndex, posX, posY))

                if showProfiler:  # Timing debug
                    Tind += 1
                    T += [time.perf_counter()]
                    print('    T(%d)-T(%d)=%f ms - runDetect end' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                # Copy values (to keep floating precision)
                pos2D = np.array([posX, posY])

                # Prepare crop range
                Xmin = max(int(posX) - self.cropSize[0] // 2, 0)
                Xmax = min(Xmin + self.cropSize[0], imgWidth)
                Xmin = Xmax - self.cropSize[0]
                Ymin = max(int(posY) - self.cropSize[1] // 2, 0)
                Ymax = min(Ymin + self.cropSize[1], imgHeight)
                Ymin = Ymax - self.cropSize[1]
                self.imgCropULs[camInd] = [Xmin, Ymin]  # ints

                if self.runDLC.value or self.imgMode.value == 'crop':
                    # Prepare cropped image
                    if self.useExclMask:
                        imgMasked = np.zeros((imgHeight, imgWidth, 3), 'uint8')
                        imgMasked[:,:,0] = cv2.bitwise_and(img[:,:,0], imgMask)
                        imgMasked[:,:,1] = cv2.bitwise_and(img[:,:,1], imgMask)
                        imgMasked[:,:,2] = cv2.bitwise_and(img[:,:,2], imgMask)
                        self.imgCrops[camInd] = np.copy(imgMasked[Ymin:Ymax, Xmin:Xmax, :])
                        # self.imgCrops[camInd][:,:,0] = cv2.bitwise_and(img[:,:,0], imgMask)[Ymin:Ymax, Xmin:Xmax]
                        # self.imgCrops[camInd][:,:,1] = cv2.bitwise_and(img[:,:,1], imgMask)[Ymin:Ymax, Xmin:Xmax]
                        # self.imgCrops[camInd][:,:,2] = cv2.bitwise_and(img[:,:,2], imgMask)[Ymin:Ymax, Xmin:Xmax]
                    else:
                        self.imgCrops[camInd] = np.copy(img[Ymin:Ymax, Xmin:Xmax, :])
                    self.log.LogText(3, 'VideoProcessing(%d): image %03d cropped, upper-left corner (%d, %d)' % (camNb, imgIndex, Xmin, Ymin))

                if showProfiler:       # Timing debug
                    Tind += 1
                    T += [time.perf_counter()]
                    print('    T(%d)-T(%d)=%f ms - copy image to imgCrops' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000))

            # Stores into shared variables
            self.pos2Ds[camInd] = pos2D.copy()
            self.imgIndexes[camInd] = imgIndex

            # Stores monitoring images
            if self.imgMode.value == 'full':
                img = cv2.resize(img, (imgWidth // 2, imgHeight // 2))              # Resize before sharing

                if showProfiler:  # Timing debug
                    Tind += 1
                    T += [time.perf_counter()]
                    print('    T(%d)-T(%d)=%f ms - resize image for imgMonits' % (Tind, Tind-1, (T[Tind]-T[Tind-1]) * 1000))

                self.imgMonits[camInd] = img.copy()
            elif self.imgMode.value == 'diff':
                imgDiff = cv2.cvtColor(imgDiff, cv2.COLOR_GRAY2BGR)
                imgDiff = cv2.resize(imgDiff, (imgWidth // 2, imgHeight // 2))      # Convert to color and resize before sharing
                self.imgMonits[camInd] = imgDiff.copy()
            elif self.imgMode.value == 'thresh':
                imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)
                imgThresh = cv2.resize(imgThresh, (imgWidth // 2, imgHeight // 2))  # Convert to color and resize before sharing
                self.imgMonits[camInd] = imgThresh.copy()
            elif self.imgMode.value == 'morph':
                imgMorph = cv2.cvtColor(imgMorph, cv2.COLOR_GRAY2BGR)
                imgMorph = cv2.resize(imgMorph, (imgWidth // 2, imgHeight // 2))    # Convert to color and resize before sharing
                self.imgMonits[camInd] = imgMorph.copy()

            if showProfiler:       # Timing debug
                Tind += 1
                T += [time.perf_counter()]
                print('    T(%d)-T(%d)=%f ms - copy image to imgMonits' % (Tind, Tind-1, (T[Tind]-T[Tind-1])*1000))

            # Show performance (missed frames each second and framerate)
            if showProfiler or showPerfs:
                missed += (xiImg.nframe - nframePrev) // self.frameStep - 1     # Missed frames
                t1 = time.perf_counter()
                missedCount += missed
                if t1 - tm > 1:   # Every one second, estimates missed frames
                    missedPercent = 100 * missedCount / self.framerate
                    self.log.LogText(2, 'VideoProcessing(%d): missed=%.f/%d frames (%.1f%%)' % (camNb, missedCount, self.framerate, missedPercent))
                    missedCount = 0
                    tm = t1
                fps = 1.0 / (t1-t0)
                t0 = t1
                missed = 0
                self.log.LogText(3, 'VideoProcessing(%d): (%d) framerate=%.1f fps' % (camNb, imgIndex, fps))

            nframePrev = xiImg.nframe


    # Run DeepLabCut inferences on cropped images (PROCESS)
    def RunDLC(self):
        """Runs DLC inferences in a separate process (starting 4 threads, one per camera)"""

        self.log.LogText(1, 'RunDLC() process running (PID: %d)' % mp.current_process().pid)

        self.DLCrunning.value = False

        # DLC neural network initialization
        if True:
            import tensorflow as tf
            from dlclive import DLCLive

            # DLC inferred keypoints vars
            keyCyclopInds = [list(self.keyNames).index(keyName) for keyName in self.keyCyclop]

            # Limite l'expansion abusive de la mémoire par Tensor flow :
            #   - entre 0 et 1, fraction de la mémoire de la carte graphique
            #   - si le même réseau pour tous les process, c'est la même mémoire pour tous, sinon pour chaque réseau
            #   - ne fonctionne pas avec TF RT
            GPUoptions = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.gpuMemoryfrac)
            config = tf.compat.v1.ConfigProto(gpu_options=GPUoptions)

            # Pour TF 1.5 :
            # GPUoptions = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            # config = tf.ConfigProto(GPUoptions=GPUoptions)

            # Choix du réseau (parmis ceux entrainés) :
            # - dossier du réseau
            # - model_type : base (avec GPU), tensorrt (plus performant, avec GPU), tflite (CPU, pas recommandé)
            # - tf_config : configuration définie juste au dessus (limitation mémoire)
            # - processor : Processor() (peut être redéfini pour ajouter du filtrage prédictif par exemple)
            # - display : pour tracer des points (pas encore trouvé comment, pas utilisé)
            # dlc_live = DLCLive(self.neuralNetDir, model_type=optimization, tf_config=config, processor=Processor(), display=False)
            dlc_live = DLCLive(self.neuralNetDir, model_type=self.optimization, tf_config=config, resize=self.cropSize[0]/self.nativeCropSize[0])

            # Initialize weights with black image (0 info)
            dlc_live.init_inference(np.zeros(self.nativeCropSize))

        # Threaded DLC inference
        def DLC_inference(self, camInd, dlc_live):
            """Runs DLC inferences on camera camInd (THREAD)"""

            camNb = self.camList[camInd]
            self.log.LogText(1, 'DLC_inference(%d) thread running' % camNb)
            imgIndexPrev = -1

            while True:

                # Quits on stopRequest
                if self.stopRequest.value or self.quit.value:
                    self.log.LogText(1, 'DLC_inference(%d): stop request received, thread ending' % camNb)
                    return

                # Get img index and skips if same
                imgIndex = self.imgIndexes[camInd]
                if self.imgIndexes[camInd] == imgIndexPrev:
                    continue

                # Gets current img
                # imgDLC = np.copy(self.imgCrops[camInd])          # Inferences on cropped image (basic detection or dynamic crop)
                cropULs = self.imgCropULs[camInd].copy()

                # Convert to DLC format (values between 0 and 1)
                imgDLC = np.array(self.imgCrops[camInd], 'float32') / 255

                # Run DLC pose inference
                inferredPos = dlc_live.get_pose(imgDLC)

                # Rescale according to native crop size and adds UL coordinates and stores in shared variable
                for keyInd in range(nKeys):
                    if inferredPos[keyInd, 2] >= self.pThresh:
                        if self.runDetect:
                            # Rescale
                            # inferredPos[keyInd, 0] *= self.nativeCropSize[0] / self.cropSize[0]
                            # inferredPos[keyInd, 1] *= self.nativeCropSize[1] / self.cropSize[1]
                            # Adds upper-left corner to inferred keypoints (for basic detection)
                            inferredPos[keyInd, 0] += cropULs[0]
                            inferredPos[keyInd, 1] += cropULs[1]
                    else:
                        inferredPos[keyInd, 0:2] = -np.ones(2)

                # Stores in shared variable for further triangulation and results saving
                self.pos2Ds_DLC[camInd] = np.copy(inferredPos)
                self.imgIndexes_DLC[camInd] = imgIndex

                # Computes cyclop
                keyAvgValidIndexes = [keyInd for keyInd in keyCyclopInds if self.pos2Ds_DLC[camInd][keyInd, 2] >= self.pThreshCyclop]
                if len(keyAvgValidIndexes) > 1:      # Minimum of 2 keypoints in the list must be valid
                    self.pos2D_cyclop[camInd] = np.average(self.pos2Ds_DLC[camInd][keyAvgValidIndexes, :], axis=0)
                else:
                    self.pos2D_cyclop[camInd] = -np.ones(3)
                    # self.log.LogText(4, 'DLC_Inference(%d): inference quality on imgIndex=%d is too low to return a valid cyclop' % (camInd, imgIndex))

        # Run DLC inference threads, one for each camera
        DLC_threads = [None] * self.camCount
        for camInd in range(self.camCount):
            DLC_threads[camInd] = threading.Thread(target=DLC_inference, args=(self, camInd, dlc_live))
            DLC_threads[camInd].start()

        self.DLCrunning.value = True
        self.log.LogText(1, 'RunDLC: all DLC_inference threads have been launched, quitting')
        return

    def Triangulation(self):
        """Triangulates across valid 2D positions (PROCESS)"""

        self.log.LogText(1, 'Triangulation() process running (PID: %d)' % mp.current_process().pid)

        # Loads camera combinations to use for triangulation
        filename = 'Settings/camCombUse.txt'
        if not os.path.isfile(filename):
            self.log.LogText(1, 'Error: cannot find file \'%s\' needed for triangulation' % filename)
            return -1
        camCombUseDict = {}
        with open(filename, 'r') as fSet:
            lines = fSet.readlines()
            for line in lines[1:]:  # Skip first line with header
                values = re.split('\t+', line)
                if len(values) < 2 or values[0][0] == '#':
                    continue
                camCombUseDict[values[0]] = dict(use=values[1], refCam=int(values[2]))
                camCombUseDict[values[0]]['pMatrix_VirtToReal'] = np.load(self.calibDir + 'Ref cam%d/Pmatrix_virt2real_%s.npy' % (camCombUseDict[values[0]]['refCam'], camCombUseDict[values[0]]['use']))

        # Load projection matrices of each camera in each reference camera
        pMatrices = []
        for camRefInd in self.camList:
            pMatrices += [[np.load(self.calibDir + 'Ref cam%d/Pmatrix_cam%d.npy' % (camRefInd,camInd)) for camInd in self.camList]]

        # DLC inferred keypoints vars
        if self.getCyclop3D:
            keyCyclopInds = [list(self.keyNames).index(keyName) for keyName in self.keyCyclop]

        # Function to triangulate pos2Ds (from 2 to 4 cameras)
        def Triangulate(pos2Ds_copy):

            hasProbaDLC = len(pos2Ds_copy[0]) == 3             # True if DLC inferred keypoints (or cyclop) with proba
            checkProbaDLC = self.checkProbaDLC and hasProbaDLC       # Sanity check

            # Determines valid camera combination (and string)
            validCamCombStr = ''
            for camNb in range(self.camCount):
                if camNb in self.camList:
                    camInd = camIndList[camNb]    # Gets camInd corresponding to camNb
                    # if not np.any(pos2Ds_copy[camInd]):         # Check for non 0
                    # if np.all(pos2Ds_copy[camInd] == -1):       # Check for non -1
                    if np.all(pos2Ds_copy[camInd][:2] == -1) or (checkProbaDLC and pos2Ds_copy[camInd, 2] < self.pThresh):
                            validCamCombStr += '_'
                    else:
                        validCamCombStr += str(camNb)
                else:
                    validCamCombStr += '_'

            # Skips if validCamComb not available (should occur only for '___')
            if validCamCombStr not in camCombUseDict.keys():
                self.log.LogText(4, 'Triangulation (imgIndex=%d): not enough valid cameras (%s) with good probability' % (self.imgIndexes[0], validCamCombStr))
                return -np.ones(3), []

            # Points to corresponding camComb to use and refCam and pMatrix_VirtToReal
            camCombStrUse = camCombUseDict[validCamCombStr]['use']
            camCombRefInd = camCombUseDict[validCamCombStr]['refCam']
            pMatrix_VirtToReal = camCombUseDict[validCamCombStr]['pMatrix_VirtToReal']

            # Get cameras in the valid camera combination
            camsUsed = [camInd for camInd in self.camList if str(camInd) in camCombStrUse]

            # Builds big matrix with all cameras involved
            A = np.zeros((len(camsUsed) * 2, 4))
            pos = 0
            for camNb in camsUsed:
                camInd = camIndList[camNb]         # Gets camInd corresponding to camNb
                x = pos2Ds_copy[camInd][0]
                y = pos2Ds_copy[camInd][1]
                P = pMatrices[camCombRefInd][camNb]
                A[2 * pos] = x * P[2] - P[0]
                A[2 * pos + 1] = y * P[2] - P[1]
                pos += 1

            # Search for singular values of A, see https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
            u, d, vt = np.linalg.svd(A)

            # Computes pos3D in cameras virtual reference frame (3*1 matrix)
            pos3Dvirt = vt[-1, 0:3] / vt[-1, 3]

            # Passes pos3D from virtual to real reference frame
            pos3Dreal = np.dot(pMatrix_VirtToReal[:, :3], pos3Dvirt)
            pos3Dreal += pMatrix_VirtToReal[:, 3:4].T[0]

            self.log.LogText(4, 'Triangulation (imgIndex=%d): %s (%s)' % (self.imgIndexes[0], pos3Dreal, camsUsed))
            return pos3Dreal, camsUsed

        def OutOfCave(pos3D_copy, XYlim=52, Zmin=-2, Zmax=70):
            """Check if a triangulated pos3D is out of the cave (with tolerance)"""

            if self.excludeOutOfCave:
                return np.abs(pos3D_copy[0]) > XYlim or np.abs(pos3D_copy[1]) > XYlim or pos3D_copy[2] > Zmax or pos3D_copy[2] < Zmin  # In cm
            else:
                return False

        # Starts connection with Rendering PC
        UDPServerSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Initializes sliding window of filter3D
        if self.filter3D != 0:
            pos3Dslide = np.zeros((3, self.filter3Dsize))
        imgIndexesPrev = -1
        imgIndexesPrev_DLC = -1

        # Gets reverse of camList to get camInd from a given camNb
        camIndList = [-1] * self.camCount
        for camInd in range(self.camCount):
            camIndList[self.camList[camInd]] = camInd

        updatePosUDP = False
        while True:

            if self.stopRequest.value:
                self.log.LogText(1, 'Triangulation: stop request received, process ending')
                UDPServerSocket.close()
                return

            # Triangulates detected 2D positions
            if self.runDetect.value:

                # Check if one of the indexes has changed (commented: and if all indexes are the same)
                if imgIndexesPrev != self.imgIndexes[0]:     # and all([self.imgIndexes[0] == self.imgIndexes[camInd] for camInd in range(1, self.camCount)]):
                    imgIndexesPrev = self.imgIndexes[0]

                    # Get copies of current 2D position list and image index
                    pos2Ds_copy = list(pos2Ds).copy()
                    imgIndex0 = self.imgIndexes[0]

                    self.log.LogText(3, 'Triangulation (imgIndexes=%d): processing detected 2D positions' % imgIndexesPrev)

                    # Triangulates detected position and stores
                    pos3D_tmp, validCams = Triangulate(pos2Ds_copy)
                    if OutOfCave(pos3D_tmp):
                        # Invalid, sets to -1
                        pos3D_tmp = -np.ones(3)
                    else:
                        # Takes the index of first camera
                        self.imgIndexPos3D.value = imgIndex0
                        if not self.useCyclop.value:
                            posUDP = pos3D_tmp
                            vcUDP = validCams
                            updatePosUDP = True
                    self.pos3D[0] = pos3D_tmp

            # Triangulates DLC inferred keypoints
            if self.runDLC.value:

                # Check if one of the indexes has changed and if all indexes are the same
                if imgIndexesPrev_DLC != self.imgIndexes_DLC[0] and all([self.imgIndexes_DLC[0] == self.imgIndexes_DLC[camInd] for camInd in range(1, self.camCount)]):
                    imgIndexesPrev_DLC = self.imgIndexes_DLC[0]

                    # Get copies of current 2D DLC keypoint position lists and image index
                    pos2Ds_DLC_copy = np.copy(pos2D_DLC)
                    pos2D_cyclop_copy = np.copy(pos2D_cyclop)
                    imgIndex0_DLC = self.imgIndexes_DLC[0]

                    self.log.LogText(3, 'Triangulation (imgIndexes_DLC=%d): processing inferred DLC keypoints' % imgIndexesPrev_DLC)

                    # Triangulates DLC keypoints
                    pos3Ds_DLC_tmp = np.zeros((nKeys, 4))           # Temporary triangulated keypoints (candidates)
                    for keyInd in range(nKeys):
                        # Triangulates keypoint keyInd
                        pos3Ds_DLC_tmp[keyInd, :3], validCams = Triangulate(pos2Ds_DLC_copy[:self.camCount, keyInd])
                        if OutOfCave(pos3Ds_DLC_tmp[keyInd, :3]):
                            # Invalid, sets to -1
                            pos3Ds_DLC_tmp[keyInd, :] = -np.ones(4)
                        else:
                            # Computes average proba of valid 2D keypoints
                            pos3Ds_DLC_tmp[keyInd, 3] = np.mean(pos2Ds_DLC_copy[validCams, keyInd, 2])
                    self.pos3Ds_DLC[0] = pos3Ds_DLC_tmp                      # Stores output in manager
                    self.imgIndexPos3D_DLC.value = imgIndex0_DLC             # Takes first camera index

                    self.log.LogText(3, 'Triangulation (imgIndexes_DLC=%d): processing cyclops' % imgIndexesPrev_DLC)

                    # Triangulates cyclop
                    pos3D_cyclop_tmp = np.zeros(4)                      # Temporary cyclop (candidate)
                    if self.getCyclop3D:
                        # Computes average keypoint positions (with minimal validity)
                        keyAvgValidIndexes = [keyInd for keyInd in keyCyclopInds if self.pos3Ds_DLC[0][keyInd, 3] >= self.pThreshCyclop]
                        if len(keyAvgValidIndexes) >= self.nKeysCyclop:       # Minimum valid keypoints in the list
                            pos3D_cyclop_tmp = np.average(self.pos3Ds_DLC[0][keyAvgValidIndexes, :], axis=0)
                        else:
                            pos3D_cyclop_tmp = -np.ones(4)              # Invalid
                            self.log.LogText(4, 'Inference quality at frame %d is too low to return a valid cyclop' % imgIndex0)
                    else:
                        # Triangulates 2D cyclops
                        if np.any(pos2D_cyclop_copy[:self.camCount, :] != -1):
                            pos3D_cyclop_tmp[:3], validCams = Triangulate(pos2D_cyclop_copy[:self.camCount])
                            if OutOfCave(pos3D_cyclop_tmp[:3]):
                                pos3D_cyclop_tmp = -np.ones(4)          # Invalid
                            else:
                                # Computes average proba of valid 2D cyclops
                                pos3D_cyclop_tmp[3] = np.mean(pos2D_cyclop_copy[validCams, 2])
                                if self.useCyclop.value:
                                    posUDP = pos3D_cyclop_tmp[:3]
                                    vcUDP = validCams
                                    updatePosUDP = True
                        else:
                            pos3D_cyclop_tmp = -np.ones(4)                  # One point is missing for triangulation
                    self.pos3D_cyclop[0] = pos3D_cyclop_tmp

            # Send 3D position to rendering via UDP (only valid when updatePosUDP is True)
            if self.sendPos3D.value and updatePosUDP:

                # Filter on pos 3D to be sent : mean or median over sliding window
                if self.filter3D == 1:      # Mean over sliding window
                    pos3Dslide = np.concatenate((pos3Dslide[:, 1:], posUDP.reshape(3, 1)), 1)
                    posUDP = np.mean(pos3Dslide, 1)
                elif self.filter3D == 2:      # Median over sliding window
                    pos3Dslide = np.concatenate((pos3Dslide[:, 1:], posUDP.reshape(3, 1)), 1)
                    posUDP = np.median(pos3Dslide, 1)

                # Send data
                message = '%.3f,%.3f,%.3f' % tuple(posUDP)
                UDPServerSocket.sendto(message.encode(), UDPserverRendering)
                self.log.LogText(3, 'Triangulation: message sent to Rendering \'%s\' (camsUsed=%s, imgIndexes=%d)' % (message, vcUDP, imgIndexesPrev))

                updatePosUDP = False

    # TCP server receiving commands from Rendering PC (THREAD)
    def TCPserver(self, cmdSeparator='\t'):
        """TCP server receiving commands from Rendering PC and GUI(THREAD)"""

        self.log.LogText(1, 'TCPserver() thread running (waiting for Rendering commands)')

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serverSocket:

            # Binds TCP server to the correct port
            while True:
                try:
                    serverSocket.bind(TCPserverTracking)
                except socket.error as errorMsg:
                    self.log.LogText(2, 'TCPserver: could not bind with server address, retrying in 1s')
                    time.sleep(1)
                    continue
                break

            self.log.LogText(2, 'TCPserver: up and running waiting for connections')
            self.serverRunning.value = True         # Creates GUI once TCP server is running

            while True:

                # Waits for client to connect
                self.log.LogText(2, 'TCPserver: listening on %s...' % str(TCPserverTracking))
                serverSocket.listen()
                try:
                    clientSocket, clientAddress = serverSocket.accept()
                except socket.error as errorMsg:
                    self.log.LogText(2, 'TCPserver: connection error: %s' % errorMsg)
                    serverSocket.close()
                    self.quit.value = True
                    return 0
                self.log.LogText(2, 'TCPserver: connected to client %s' % str(clientAddress))

                with clientSocket:

                    # Reads message (one chunks)
                    try:
                        msg = clientSocket.recv(TCPpacketSize)      # Reads next chunk
                    except socket.error as errorMsg:
                        self.log.LogText(2, 'TCPserver: reception error: %s, ignoring' % errorMsg)
                        msg = b''
                    if not msg:         # Empty message: end of transmission
                        clientSocket.close()  # Close client connection
                        self.log.LogText(2, 'TCPserver: closing client connection')
                        continue

                    # Acknowledge reception
                    # clientSocket.sendall(('TCPserver on Tracking: message received').encode())

                    # Reads and processes list of commands 
                    cmdList = msg.decode('utf-8').split(cmdSeparator)
                    for cmd in cmdList:
                        if not len(cmd): break

                        self.log.LogText(3, 'TCPserver: command=\'%s\'' % cmd)
                        # Process start, stop and quit commands (dispatched to other threads)
                        if cmd == 'startExperiment':
                            self.log.LogText(2, 'TCPserver: \'startExperiment\' received from rendering')
                            self.CheckSettings(atLoad=False)        # Check if current settings are ok
                            self.dataRecording.value = False
                            self.startRequest.value = True
                        elif cmd == 'endExperiment':
                            self.log.LogText(2, 'TCPserver: \'endExperiment\' received from rendering')
                            self.dataRecording.value = False
                            self.stopRequest.value = True
                            if self.recoboxRunning.value:
                                self.SendCommandTCPRecobox('LED OFF', 0)
                                self.SendCommandTCPRecobox('BUZZER OFF', 0)
                        elif cmd == 'startTrial':
                            if len(self.expID.value) > 0 and len(self.subjectID.value) > 0 and len(self.trialID.value) > 0 and len(self.condID.value) > 0:
                                self.log.LogText(2, 'TCPserver: \'startTrial\' received from rendering with :')
                                self.log.LogText(2, '  expID=\'%s\'' % self.expID.value)
                                self.log.LogText(2, '  subjectID=\'%s\'' % self.subjectID.value)
                                self.log.LogText(2, '  trialID=\'%s\'' % self.trialID.value)
                                self.log.LogText(2, '  condID=\'%s\'' % self.condID.value)
                                self.log.LogText(2, '  saveResults=%s' % (self.saveResults.value==True))
                                self.log.LogText(2, '  recSound=%s' % (self.recSound.value==True))
                                # Sync image indexes between jetsons
                                self.syncRequest[:self.camCount] = [True] * self.camCount
                                while self.syncRequest[:self.camCount] != [False] * self.camCount: pass
                                # Start recording data
                                if self.saveResults.value:
                                    self.dataRecording.value = True
                                    self.log.LogText(2, 'TCPserver: saveResults activated')
                                    self.videoReady[:self.camCount] = [False] * self.camCount
                                self.trialStarted.value = True
                                self.eventString.value = 'startTrial'
                            else:
                                self.log.LogText(2, 'TCPserver: \'startTrial\' received but ignored, send expID, subjectID, trialID, condID first')
                        elif cmd == 'startAnimation':
                            self.log.LogText(2, 'TCPserver: \'startAnimation\' received from rendering')
                            self.eventString.value = 'startAnimation'
                        elif cmd == 'startWaiting':
                            self.log.LogText(2, 'TCPserver: \'startWaiting\' received from rendering')
                            self.eventString.value = 'startWaiting'
                        elif cmd == 'endWaiting':
                            self.log.LogText(2, 'TCPserver: \'endWaiting\' received from rendering')
                            self.eventString.value = 'endWaiting'
                        elif cmd == 'endTrial':
                            self.log.LogText(2, 'TCPserver: \'endTrial\' received from rendering')
                            self.eventString.value = 'endTrial'
                            self.dataRecording.value = False
                            self.trialStarted.value = False
                            if self.recoboxRunning.value:
                                self.SendCommandTCPRecobox('LED OFF', 0)
                                self.SendCommandTCPRecobox('BUZZER OFF', 0)
                        elif cmd == 'quit':
                            self.log.LogText(2, 'TCPserver: \'quit\' received from rendering')
                            self.quit.value = True
                            self.stopRequest.value = True
                            if self.recoboxRunning.value:
                                self.SendCommandTCPRecobox('LED OFF', 0)
                                self.SendCommandTCPRecobox('BUZZER OFF', 0)
                            clientSocket.close()
                            serverSocket.close()
                            return 0

                        # Command for the Recobox
                        elif cmd == 'LED R ON':
                            self.SendCommandTCPRecobox('LED R ON', 0)
                            self.log.LogText(2, 'TCPserver: \'LED R ON\' received from rendering')
                        elif cmd == 'LED R BLINK':
                            self.SendCommandTCPRecobox('LED R BLINK', 0)
                            self.log.LogText(2, 'TCPserver: \'LED R BLINK\' received from rendering')
                        elif cmd == 'LED G ON':
                            self.SendCommandTCPRecobox('LED G ON', 0)
                            self.log.LogText(2, 'TCPserver: \'LED G ON\' received from rendering')
                        elif cmd == 'LED G BLINK':
                            self.SendCommandTCPRecobox('LED G BLINK', 0)
                            self.log.LogText(2, 'TCPserver: \'LED G BLINK\' received from rendering')
                        elif cmd == 'LED Y ON':
                            self.SendCommandTCPRecobox('LED Y ON', 0)
                            self.log.LogText(2, 'TCPserver: \'LED Y ON\' received from rendering')
                        elif cmd == 'LED Y BLINK':
                            self.SendCommandTCPRecobox('LED Y BLINK', 0)
                            self.log.LogText(2, 'TCPserver: \'LED Y BLINK\' received from rendering')
                        elif cmd == 'LED OFF':
                            self.SendCommandTCPRecobox('LED OFF', 0)
                            self.log.LogText(2, 'TCPserver: \'LED OFF\' received from rendering')
                        elif cmd == 'BUZZER ON':
                            self.SendCommandTCPRecobox('BUZZER ON', 0)
                            self.log.LogText(2, 'TCPserver: \'BUZZER ON\' received from rendering')
                        elif cmd == 'BUZZER BLINK':
                            self.SendCommandTCPRecobox('BUZZER BLINK', 0)
                            self.log.LogText(2, 'TCPserver: \'BUZZER BLINK\' received from rendering')
                        elif cmd == 'BUZZER OFF':
                            self.SendCommandTCPRecobox('BUZZER OFF', 0)
                            self.log.LogText(2, 'TCPserver: \'BUZZER OFF\' received from rendering')
                        elif cmd == 'REWARD':
                            self.SendCommandTCPRecobox('REWARD', 0)
                            self.log.LogText(2, 'TCPserver: \'REWARD\' received from rendering')

                        # From here on parameters cannot be set if trial started ongoing
                        elif self.trialStarted.value:
                            self.log.LogText(3, 'TCPserver: trial started, command \'%s\' will be ignored' % cmd)

                        # Ask for new reference images
                        elif cmd == 'newRef':
                            self.log.LogText(2, 'TCPserver: \'newRef\' received from rendering')
                            self.newRefRequest[:self.camCount] = [True] * self.camCount

                        elif cmd == 'REFILL':
                            self.SendCommandTCPRecobox('REFILL', 0)
                            self.log.LogText(2, 'TCPserver: \'REFILL\' received from rendering')
                        elif cmd == 'CLEAN':
                            self.SendCommandTCPRecobox('CLEAN', 0)
                            self.log.LogText(2, 'TCPserver: \'CLEAN\' received from rendering')

                        # Reload settings
                        elif cmd == 'loadSettings':
                            self.log.LogText(2, 'TCPserver: \'loadSettings\' received from rendering')
                            self.LoadSettings()
                            self.CheckSettings()

                        # Loads experiment parameters
                        elif 'expID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: expID=\'%s\'' % val)
                            exec('self.expID.value=\'%s\'' % val)
                        elif 'subjectID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: subjectID=\'%s\'' % val)
                            exec('self.subjectID.value=\'%s\'' % val)
                        elif 'trialID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: trialID=\'%s\'' % val)
                            exec('self.trialID.value=\'%s\'' % val)
                        elif 'condID' in cmd:
                            val = cmd.split(sep='=')[1]
                            self.log.LogText(2, 'TCPserver: condID=\'%s\'' % val)
                            exec('self.condID.value=\'%s\'' % val)

                        # Unknown command
                        else:
                            self.log.LogText(3, 'TCPserver: command not understood \'%s\'' % cmd)

    # Save results: detected center pos2D and DLC keypoints (for all cameras), and pos3D triangulations (PROCESS)
    def SaveResults(self, framerate, maxDuration=3600):

        self.log.LogText(1, 'SaveResults process running (PID: %d)' % mp.current_process().pid)

        dataRecordingPrev = False               # Stores previous flag state

        # Prepare new results2D array dt
        dt2D = [('time', 'f8')]
        for camInd in range(self.camCount):
            dt2D.extend([('imgIndex_cam%d' % camInd, 'u4'), ('pos(UL)_cam%d' % camInd, 'u2', 2)])
            if self.runDetect.value:
                dt2D.extend([('pos(0)_cam%d' % camInd, 'f8', 2)])
            if self.runDLC.value:
                dt2D.extend([('pos(Cyclop)_cam%d' % camInd, 'f8', 2), ('proba(Cyclop)_cam%d' % camInd, 'f8')])
                for keyName in self.keyNames:
                    dt2D.extend([('pos(%s)_cam%d' % (keyName, camInd), 'f8', 2), ('proba(%s)_cam%d' % (keyName, camInd), 'f8')])
        dt2D = np.dtype(dt2D)

        # Prepare new results3D array dt
        dt3D = [('time', 'f8'), ('imgIndex', 'u4')]  # for new res3D array
        if self.runDetect.value:
            dt3D.extend([('pos(0)', 'f8', 3)])
            if not self.runDLC.value and self.getVelocity:
                dt3D.extend([('vel(0)', 'f8', 3), ('velNorm(0)', 'f8')])
        if self.runDLC.value:
            dt3D.extend([('pos(Cyclop)', 'f8', 3), ('proba(Cyclop)', 'f8')])
            for keyName in self.keyNames:
                dt3D.extend([('pos(%s)' % keyName, 'f8', 3), ('proba(%s)' % keyName, 'f8')])
            if self.getVelocity:
                dt3D.extend([('vel(Cyclop)', 'f8', 3), ('velNorm(Cyclop)', 'f8')])
            if self.getMotionDir:
                dt3D.extend([('motionDir', 'i1')])
            if self.getGazeDir:
                dt3D.extend([('gazeDir', 'f8', 3)])

                # DLC inferred keypoints vars
                keyGazeDirInds = [list(self.keyNames).index(keyName) for keyName in self.keyGazeDir]

        dt3D = np.dtype(dt3D)

        # Prepare new resultsEvents array dt
        dtEvents = np.dtype([('time', 'f8'), ('imgIndex', 'u4'), ('event', 'U30')])

        while True:

            # Quits on stopRequest
            if self.stopRequest.value:
                self.dataRecording.value = False     # Sends signal to stop data recording (in case)
                if not dataRecordingPrev:   # Quits when no longer recording
                    self.log.LogText(1, 'SaveResults: stop request received, process ending')
                    return

            if self.dataRecording.value:

                if not dataRecordingPrev:
                    # At trial start: initializes arrays
                    self.log.LogText(2, 'SaveResults: preparing arrays (flag dataRecording=True)')

                    # Creates paths if not existing
                    path = self.resultsDir + '%s/%s/' % (self.expID.value, self.subjectID.value)
                    resultsFile = 'Trial-%s_Cond-%s' % (self.trialID.value, self.condID.value)
                    try:
                        os.makedirs(path)
                    except FileExistsError:
                        pass

                    # Prepare arrays (limited to maxDuration of recording)
                    if self.runDetect.value or self.runDLC.value:
                        res2D = np.recarray((framerate * maxDuration,), dtype=dt2D)
                        res2D['time'] = -1.0
                        if self.triangulate.value:
                            res3D = np.recarray((framerate * maxDuration,), dtype=dt3D)
                            res3D['time'] = -1.0
                    resEvents = np.recarray((framerate * maxDuration,), dtype=dtEvents)
                    resEvents['time'] = -1.0
                    self.eventString.value = 'startTrial'

                    # Store initial index of cam0 for time column and rowIndex computations
                    indexStart = 0      # was self.imgIndexes[0] or pos2DimgIndexes[0]
                    imgIndexesPrev = [-1] * self.camCount
                    imgIndexesPrev_DLC = [-1] * self.camCount
                    imgIndexPos3DPrev = -1
                    imgIndexPos3DPrev_DLC = -1

                    # Starts recording sound
                    if self.recSound.value:
                        # Sound parameters
                        sample_rate = 44100
                        bits_per_sample = 16
                        chunk_size = 1024
                        audio_format = pyaudio.paInt16
                        channels = 1

                        # Recording callback
                        def callback(in_data, frame_count, time_info, status):
                            wav_file.writeframes(in_data)
                            return None, pyaudio.paContinue
                     
                        # Open the wave file for writing
                        filename_wav = path + resultsFile + '_sound.wav'
                        wav_file = wave.open(filename_wav, 'wb')
                        wav_file.setnchannels(channels)
                        wav_file.setsampwidth(bits_per_sample // 8)
                        wav_file.setframerate(sample_rate)

                        # Initialize PyAudio
                        audio = pyaudio.PyAudio()

                        # Start recording audio
                        stream = audio.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size, stream_callback=callback)

                    dataRecordingPrev = True

                # During trial: records data

                # 2D data
                for camInd in range(self.camCount):
                    if self.runDetect.value:
                        if self.imgIndexes[camInd] != imgIndexesPrev[camInd]:
                            imgIndexesPrev[camInd] = self.imgIndexes[camInd]
                            fInd = int(self.imgIndexes[camInd] - indexStart)
                            res2D['time'][fInd] = fInd * 1000/self.framerate
                            res2D['imgIndex_cam%d' % camInd][fInd] = fInd
                            res2D['pos(UL)_cam%d' % camInd][fInd] = self.imgCropULs[camInd]
                            res2D['pos(0)_cam%d' % camInd][fInd] = self.pos2Ds[camInd]
                            # self.log.LogText(3, 'SaveResults: new 2D data available, updating results arrays ')
                    if self.runDLC.value:
                        if self.imgIndexes_DLC[camInd] != imgIndexesPrev_DLC[camInd]:
                            imgIndexesPrev_DLC[camInd] = self.imgIndexes_DLC[camInd]
                            fInd = int(self.imgIndexes_DLC[camInd] - indexStart)
                            for keyInd, keyName in enumerate(self.keyNames):
                                res2D['pos(%s)_cam%d' % (keyName, camInd)][fInd] = self.pos2Ds_DLC[camInd][keyInd, :2]     # X, Y (pos2D) of keypoint keyInd
                                res2D['proba(%s)_cam%d' % (keyName, camInd)][fInd] = self.pos2Ds_DLC[camInd][keyInd, 2]    # pMean of keypoint keyInd
                            res2D['pos(Cyclop)_cam%d' % camInd][fInd] = self.pos2D_cyclop[camInd][:2]                # X, Y (pos2D) of cyclop
                            res2D['proba(Cyclop)_cam%d' % camInd][fInd] = self.pos2D_cyclop[camInd][2]               # pMean of cyclop

                # 3D data
                if self.triangulate.value:
                    if self.runDetect.value:
                        if self.imgIndexPos3D.value != imgIndexPos3DPrev:
                            imgIndexPos3DPrev = self.imgIndexPos3D.value
                            fInd = int(self.imgIndexPos3D.value - indexStart)
                            res3D['time'][fInd] = fInd * 1000/self.framerate
                            res3D['imgIndex'][fInd] = fInd
                            res3D['pos(0)'][fInd] = self.pos3D[0]
                            if not self.runDLC.value and self.getVelocity and fInd > 0:
                                # Instant velocity vector and norm
                                if np.all(res3D['pos(0)'][fInd] != -1) and np.all(res3D['pos(0)'][fInd-1]) != -1 and np.all(res3D['pos(0)'][fInd-1] != 0):
                                    res3D['vel(0)'][fInd] = (res3D['pos(0)'][fInd] - res3D['pos(0)'][fInd-1]) / (res3D['time'][fInd] - res3D['time'][fInd-1])
                                    res3D['velNorm(0)'][fInd] = np.linalg.norm(res3D['vel(0)'][fInd])
                                else:
                                    res3D['vel(0)'][fInd] = -1
                                    res3D['velNorm(0)'][fInd] = -1
                            # self.log.LogText(3, 'SaveResults: new 3D data available, updating results arrays ')
                    if self.runDLC.value:
                        if self.imgIndexPos3D_DLC.value != imgIndexPos3DPrev_DLC:
                            imgIndexPos3DPrev_DLC = self.imgIndexPos3D_DLC.value
                            fInd = int(self.imgIndexPos3D_DLC.value - indexStart)
                            for keyInd, keyName in enumerate(self.keyNames):
                                res3D['pos(%s)' % keyName][fInd] = self.pos3Ds_DLC[0][keyInd, :3]     # X, Y, Z (pos3D) of keypoint keyInd
                                res3D['proba(%s)' % keyName][fInd] = self.pos3Ds_DLC[0][keyInd, 3]    # pMean of keypoint keyInd
                            res3D['pos(Cyclop)'][fInd] = self.pos3D_cyclop[0][:3]               # X, Y, Z (pos3D) of cyclop
                            res3D['proba(Cyclop)'][fInd] = self.pos3D_cyclop[0][3]              # pMean of cyclop
                            if self.getVelocity and fInd > 0:
                                # Instant velocity vector and norm
                                if np.all(res3D['pos(Cyclop)'][fInd] != -1) and np.all(res3D['pos(Cyclop)'][fInd-1] != -1):
                                    res3D['vel(Cyclop)'][fInd] = (res3D['pos(Cyclop)'][fInd] - res3D['pos(Cyclop)'][fInd-1]) / (res3D['time'][fInd] - res3D['time'][fInd-1])
                                    res3D['velNorm(Cyclop)'][fInd] = np.linalg.norm(res3D['vel(Cyclop)'][fInd])
                                else:
                                    res3D['vel(Cyclop)'][fInd] = -1
                                    res3D['velNorm(Cyclop)'][fInd] = -1
                            if self.getGazeDir:
                                # Instant gaze direction (normalized vector)
                                vStart, vEnd = self.pos3Ds_DLC[0][keyGazeDirInds, :3]
                                if np.all(vStart != -1) and np.all(vEnd != -1):
                                    v = vEnd - vStart
                                    res3D['gazeDir'][fInd] = v / np.linalg.norm(v)
                                else:
                                    res3D['gazeDir'][fInd] = -1
                            if self.getMotionDir:
                                # Direction of motion: forward=+1, backward=-1, static=0, undefined=-99
                                if np.all(res3D['vel(Cyclop)'][fInd] != -1) and np.all(res3D['gazeDir'][fInd] != -1):
                                    if np.dot(res3D['vel(Cyclop)'][fInd], res3D['gazeDir'][fInd]) > 0:
                                        res3D['motionDir'][fInd] = +1 if res3D['velNorm(Cyclop)'][fInd] > self.minFwdSpeed else 0
                                    else:
                                        res3D['motionDir'][fInd] = -1 if res3D['velNorm(Cyclop)'][fInd] > self.minBwdSpeed else 0
                                else:
                                    res3D['motionDir'][fInd] = -99

                # Events
                if self.eventString.value != '':
                    # print(self.eventString.value)
                    fInd = int(self.imgIndexes[0] - indexStart)
                    resEvents['time'][fInd] = fInd * 1000 / self.framerate
                    resEvents['imgIndex'][fInd] = fInd
                    resEvents['event'][fInd] = self.eventString.value
                    self.eventString.value = ''

            elif dataRecordingPrev:
                # At trial end: stores arrays (append writing mode)
                self.log.LogText(2, 'SaveResults: saving arrays, this can take a while... (flag dataRecording=False)')

                # General results log file
                header = ['expID', 'subjectID', 'trialID', 'condID', 'filename']
                fileName = self.resultsDir + '%s/' % self.expID.value + '%s_files.tsv' % self.expID.value
                writeHeader = not os.path.isfile(fileName)
                with open(fileName, 'a') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter='\t')
                    if writeHeader:
                        filewriter.writerow(header)
                    filewriter.writerow([self.expID.value, self.subjectID.value, self.trialID.value, self.condID.value, resultsFile])

                if self.runDetect.value or self.runDLC.value:
                    # 2D data (npy)
                    if self.runDetect.value:
                        if self.runDLC.value:
                            filename = path + resultsFile + '_pos2D+DLC2D'
                        else:
                            filename = path + resultsFile + '_pos2D'
                    else:
                        filename = path + resultsFile + '_DLC2D'
                    res2D = res2D[res2D['time'] != -1.0]
                    np.save(filename, res2D)

                    # 3D data (npy)
                    if self.triangulate.value:
                        if self.runDetect.value:
                            if self.runDLC.value:
                                filename = path + resultsFile + '_pos3D+DLC3D'
                            else:
                                filename = path + resultsFile + '_pos3D'
                        else:
                            filename = path + resultsFile + '_DLC3D'
                        res3D = res3D[res3D['time'] != -1.0]
                        np.save(filename, res3D)

                # Events (npy)
                filename = path + resultsFile + '_events'
                resEvents = resEvents[resEvents['time'] != -1.0]
                np.save(filename, resEvents)

                # Sound : stop and close the audio stream
                if self.recSound.value:
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                    wav_file.close()        # Close the wave file
                    
                    # Converts file to desired format
                    if self.audioFormat.lower() != 'wav':
                        # AudioSegment.converter = r"path\to\ffmpeg.exe"
                        audSeg = AudioSegment.from_wav(filename_wav)
                        # audSeg = audio.set_frame_rate(44100)
                        audSeg.set_frame_rate(44100)
                        filename_cvt = path + resultsFile + '_sound.' + self.audioFormat.lower()
                        audSeg.export(filename_cvt, format=self.audioFormat.lower())
                        
                        # Erase wav file
                        os.remove(filename_wav)
                    
                    # Adds to the video files
                    if self.audioInVideos:
                        
                        # Prepares audio channel (mp3 format, 192kbps)
                        filename_mp3 = path + resultsFile + '_sound.mp3'
                        audSeg.export(filename_mp3, format='mp3', bitrate='192k')
                        audio = AudioFileClip(filename_mp3)

                        for camInd in range(self.camCount):
                            
                            # Waits until video file is available
                            while not self.videoReady[camInd]: pass

                            # Reads original video
                            filename = path + resultsFile + '_cam%d.mp4' % self.camList[camInd]
                            video = VideoFileClip(filename)
                            
                            # Adds channel to video and saves
                            finalVideo = video.with_audio(audio)
                            finalVideo.write_videofile(filename)
                            
                            # Resets flag
                            self.videoReady[camInd] = False

                # Copy reference images of this trial (usefull for post-processing)
                for camInd in range(self.camCount):
                    camNb = self.camList[camInd]
                    shutil.copyfile('References/refCam%d.png' % camNb, path + resultsFile + '_cam%d_RefImg.png' % camNb)

                # If text copy requested (this can take a long time)
                if self.saveTextCopy and (self.runDetect.value or self.runDLC.value):

                    # 2D data (txt)
                    if self.runDetect.value:
                        if self.runDLC.value:
                            filename = path + resultsFile + '_pos2D+DLC2D'
                        else:
                            filename = path + resultsFile + '_pos2D'
                    else:
                        filename = path + resultsFile + '_DLC2D'
                    header2D = ('%s\t' * len(dt2D.names)) % dt2D.names
                    np.savetxt(filename + '.tsv', res2D, fmt='%s', delimiter='\t', header=header2D, comments='')
                    # res2D.tofile(filename + '.txt', sep='\n')

                    # 3D data (txt)
                    if self.triangulate.value:
                        if self.runDetect.value:
                            if self.runDLC.value:
                                filename = path + resultsFile + '_pos3D+DLC3D'
                            else:
                                filename = path + resultsFile + '_pos3D'
                        else:
                            filename = path + resultsFile + '_DLC3D'
                        header3D = ('%s\t' * len(dt3D.names)) % dt3D.names
                        np.savetxt(filename + '.tsv', res3D, fmt='%s', delimiter='\t', header=header3D, comments='')
                        # res3D.tofile(filename + '.txt', sep='\n')

                    # Events (txt)
                    filename = path + resultsFile + '_events'
                    headerEvents = ('%s\t' * len(dtEvents.names)) % dtEvents.names
                    np.savetxt(filename + '.tsv', resEvents, fmt='%s', delimiter='\t', header=headerEvents, comments='')
                    # resEvents.tofile(filename + '.txt', sep='\n')

                os.sync()       # Forces writing of cache data to disks

                self.log.LogText(2, 'SaveResults: arrays saved')

                dataRecordingPrev = False

    def SendCommandTCPRecobox(self, command, recoboxIndex=0):
        """Send commands to TCP server running on RecoBoxes"""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:

            # Connects to TCP server on Recobox
            try:
                clientSocket.connect(TCPserverRecobox[recoboxIndex])
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPRecobox: Connection error: %s' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPRecobox: Connected to TCP server')

            # Prepares binary message
            msg = str.encode(command)

            # Sends command
            try:
                clientSocket.sendall(msg)  # Crashes if server is down
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPRecobox: Sending error: %s, ignoring' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPRecobox: cmd=\'%s\' sent' % command)
            self.eventString.value = command

    # Starts the Graphic User Interface (in a separate process)
    def StartGUI(self):

        self.log.LogText(1, 'StartGUI: creating UIController object')
        myQtApp = QApplication(sys.argv)
        self.UI = UIController(self_tracking=self)
        myQtApp.exec_()


# GUI of tracking
class UIController(QWidget):

    def __init__(self, self_tracking):
        """Constructor"""

        self.log = self_tracking.log
        self.log.LogText(1, 'UIController() called')

        # UI shared controller vars
        UImode = self_tracking.settings.controller == 'UI'
        if self_tracking:
            self.expID = self_tracking.expID
            self.subjectID = self_tracking.subjectID
            self.trialID = self_tracking.subjectID
            self.condID = self_tracking.subjectID
        self.runDetect = self_tracking.runDetect
        self.runDLC = self_tracking.runDLC
        self.triangulate = self_tracking.triangulate
        self.showPos2D = self_tracking.showPos2D
        self.showDLC = self_tracking.showDLC
        self.useCyclop = self_tracking.useCyclop
        self.sendPos3D = self_tracking.sendPos3D
        self.saveResults = self_tracking.saveResults
        self.imgMode = manager.list(self.settings.imgMode)       # Image monitoring modes (2 max among full, crop, thresh, morph depending on detector)

        super().__init__()

        # Load section UIs
        prevY = self.SystemSettingsUI(posX=10, posY=0)
        winHeight = 390
        if self.UImode:
            prevY = self.ExperimentSettingsUI(posX=10, posY=prevY + 25)
            prevY = self.ExperimentControllerUI(posX=10, posY=prevY + 25)
            winHeight += 300
        if self.recoboxRunning.value:
            self.RecoboxControllerUI(posX=10, posY=prevY + 25)
            winHeight += 250

        # Camera return visual
        self.panel = QLabel(self)

        # General aspect of the window
        self.setFixedSize(260, winHeight)
        self.move(10, 10)
        self.setWindowTitle('Tracking UI')
        self.show()

        # TODO: update window if recobox is active (actually, checking recobox only at init, want to update this between exp)


    def SystemSettingsUI(self, posX, posY):
        # Section title
        self.systemSetLbl = QLabel('System settings', self)
        self.systemSetLbl.setGeometry(posX, posY, 150, 30)
        self.systemSetLbl.setStyleSheet('font-weight: bold')
        posY += 40
        # Run detection
        self.runDetectBtn = QPushButton('Detect animal', self)
        self.runDetectBtn.setGeometry(posX, posY, 120, 30)
        self.runDetectBtn.setCheckable(True)
        self.runDetectBtn.setChecked(self.runDetect.value)
        self.runDetectBtn.clicked.connect(self.RunDetect)
        # Show detected position
        self.showPos2DBtn = QPushButton('Show 2D pos', self)
        self.showPos2DBtn.setGeometry(posX + 120, posY, 120, 30)
        self.showPos2DBtn.setCheckable(True)
        self.showPos2DBtn.setChecked(self.showPos2D.value)
        self.showPos2DBtn.setEnabled(self.runDetect.value or self.runDLC.value)
        self.showPos2DBtn.clicked.connect(self.ShowPos2D)
        posY += 35
        # Run DeepLabCut
        self.runDLCBtn = QPushButton('DeepLabCut', self)
        self.runDLCBtn.setGeometry(posX, posY, 120, 30)
        self.runDLCBtn.setCheckable(True)
        self.runDLCBtn.setChecked(self.runDLC.value)
        self.runDLCBtn.clicked.connect(self.RunDLC)
        # Show DeepLabCut
        self.showDLCBtn = QPushButton('Show keypoints', self)
        self.showDLCBtn.setGeometry(posX + 120, posY, 120, 30)
        self.showDLCBtn.setCheckable(True)
        self.showDLCBtn.setEnabled(self.runDLC.value)
        self.showDLCBtn.setChecked(self.showDLC.value)
        self.showDLCBtn.clicked.connect(self.ShowDLC)
        posY += 35
        # Use cyclop checker
        self.useCyclopChk = QCheckBox('Use cyclop', self)
        self.useCyclopChk.setCheckable(True)
        self.useCyclopChk.setChecked(self.useCyclop.value)
        self.useCyclopChk.setEnabled(self.runDLC.value)
        self.useCyclopChk.setGeometry(posX + 100, posY, 180, 30)
        self.useCyclopChk.clicked.connect(self.UseCyclop)
        posY += 45
        # Run triangulation
        self.triangulateBtn = QPushButton('Triangulate', self)
        self.triangulateBtn.setGeometry(posX, posY, 120, 30)
        self.triangulateBtn.setCheckable(True)
        self.triangulateBtn.setChecked(self.triangulate.value)
        self.triangulateBtn.setEnabled(self.runDetect.value or self.runDLC.value)
        self.triangulateBtn.clicked.connect(self.Triangulate)
        # Send 3D position to Unreal
        self.sendPos3DBtn = QPushButton('Send 3D pos', self)
        self.sendPos3DBtn.setGeometry(posX + 120, posY, 120, 30)
        self.sendPos3DBtn.setCheckable(True)
        self.sendPos3DBtn.setEnabled(self.triangulate.value)
        self.sendPos3DBtn.setChecked(self.sendPos3D.value)
        self.sendPos3DBtn.clicked.connect(self.SendPos3D)
        posY += 35
        # Image Mode selection
        self.imgModeTxt = QLabel('Monitoring', self)
        self.imgModeTxt.setGeometry(posX + 20, posY, 120, 30)
        self.imgModeCombo = QComboBox(self)
        self.imgType = ['full', 'crop', 'diff', 'thresh', 'morph']
        self.imgModeCombo.addItems(self.imgType)
        self.imgModeCombo.setGeometry(posX + 150, posY, 90, 30)
        self.imgModeCombo.setCurrentIndex(self.imgType.index(self.imgMode.value))
        self.imgModeCombo.currentIndexChanged.connect(self.ImgMode)
        posY += 45
        # Load settings
        self.loadSettingsBtn = QPushButton('Reload Settings', self)
        self.loadSettingsBtn.setGeometry(posX + 50, posY, 140, 30)
        self.loadSettingsBtn.setEnabled(True)
        self.loadSettingsBtn.clicked.connect(self.LoadSettings)
        posY += 35

        # In UE controller mode, add...
        if not self.UImode:
            # New reference image button
            posY += 10
            self.newRefBtn = QPushButton('New Reference Image', self)
            self.newRefBtn.setGeometry(posX + 30, posY, 180, 30)
            self.newRefBtn.clicked.connect(self.NewRef)
            self.newRefBtn.setEnabled(True)
            posY += 40
            # Save results checker
            self.saveResultsChk = QCheckBox('Save results', self)
            self.saveResultsChk.setGeometry(posX + 100, posY, 180, 30)
            self.saveResultsChk.setCheckable(True)
            self.saveResultsChk.setChecked(self.saveResults.value)
            self.saveResultsChk.setEnabled(True)
            self.saveResultsChk.clicked.connect(self.SaveResults)
            posY += 30
            # Rec audio checker
            self.recSoundChk = QCheckBox('Record sound', self)
            self.recSoundChk.setGeometry(posX + 100, posY, 180, 30)
            self.recSoundChk.setCheckable(True)
            self.recSoundChk.setChecked(self.recSound.value)
            self.recSoundChk.setEnabled(True)
            self.recSoundChk.clicked.connect(self.RecSound)
            posY += 30

        return posY

    def UpdateSystemSettingsUI(self):

        self.runDetectBtn.setChecked(self.runDetect.value)
        self.showPos2DBtn.setChecked(self.showPos2D.value)
        self.triangulateBtn.setChecked(self.triangulate.value)
        if self.triangulate.value:
            self.sendPos3DBtn.setEnabled(True)
            self.sendPos3DBtn.setChecked(self.sendPos3D.value)
        self.runDLCBtn.setChecked(self.runDLC.value)
        if self.runDLC.value:
            self.showDLCBtn.setEnabled(True)
            self.showDLCBtn.setChecked(self.showDLC.value)
        self.imgModeCombo.setCurrentIndex(self.imgType.index(self.imgMode.value))
        self.saveResultsChk.setChecked(self.saveResults.value)
        self.recSoundChk.setChecked(self.recSound.value)

        self.log.LogText(2, 'UpdateSystemSettingsUI: done')

    def ExperimentSettingsUI(self, posX, posY):
        # Section title
        self.experimentSetLbl = QLabel('Experiment Settings', self)
        self.experimentSetLbl.setGeometry(posX, posY, 150, 30)
        self.experimentSetLbl.setStyleSheet('font-weight: bold')
        posY += 40
        # Experiment ID
        self.expID = QLabel('Experiment', self)
        self.expID.setGeometry(posX, posY, 100, 20)
        self.textExpID = QLineEdit(self)
        self.textExpID.setText(self.expID.value)
        self.textExpID.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # Subject ID
        self.subjID = QLabel('Subject', self)
        self.subjID.setGeometry(posX, posY, 100, 20)
        self.textSubjID = QLineEdit(self)
        self.textSubjID.setText(self.subjectID.value)
        self.textSubjID.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # TrialID
        self.trialID = QLabel('Trial', self)
        self.trialID.setGeometry(posX, posY, 100, 20)
        self.textTrialID = QLineEdit(self)
        self.textTrialID.setText(self.trialID.value)
        self.textTrialID.setGeometry(posX + 90, posY, 150, 20)
        posY += 25
        # ConditionID
        self.condID = QLabel('Condition', self)
        self.condID.setGeometry(posX, posY, 100, 20)
        self.textCondID = QLineEdit(self)
        self.textCondID.setText(self.condID.value)
        self.textCondID.setGeometry(posX + 90, posY, 150, 20)
        posY += 30

        return posY

    def UpdateExperimentSettingsUI(self):

        self.textExpID.setText(self.expID.value)
        self.textSubjID.setText(self.subjectID.value)
        self.textTrialID.setText(self.trialID.value)
        self.textCondID.setText(self.condID.value)
        self.saveResultsChk.setChecked(self.saveResults.value)
        self.recSoundChk.setChecked(self.recSound.value)

        self.log.LogText(2, 'UpdateExperimentSettingsUI: done')

    def ExperimentControllerUI(self, posX, posY):
        # Section title
        self.controllerLbl = QLabel('Experiment Controller', self)
        self.controllerLbl.setGeometry(posX, posY, 150, 30)
        self.controllerLbl.setStyleSheet('font-weight: bold')
        posY += 30
        # Start experiment
        self.startExpBtn = QPushButton('Start Experiment', self)
        self.startExpBtn.setGeometry(posX, posY, 120, 30)
        self.startExpBtn.setCheckable(True)
        self.startExpBtn.setEnabled(True)
        self.startExpBtn.clicked.connect(self.StartExperiment)
        # End experiment
        self.endExpBtn = QPushButton('End Experiment', self)
        self.endExpBtn.setGeometry(posX + 120, posY, 120, 30)
        self.endExpBtn.setCheckable(True)
        self.endExpBtn.setEnabled(False)
        self.endExpBtn.clicked.connect(self.EndExperiment)
        posY += 35
        # Start trial
        self.startTrialBtn = QPushButton('Start Trial', self)
        self.startTrialBtn.setGeometry(posX, posY, 120, 30)
        self.startTrialBtn.setEnabled(False)
        self.startTrialBtn.clicked.connect(self.StartTrial)
        # End trial
        self.endTrialBtn = QPushButton('End Trial', self)
        self.endTrialBtn.setGeometry(posX + 120, posY, 120, 30)
        self.endTrialBtn.setEnabled(False)
        self.endTrialBtn.clicked.connect(self.EndTrial)
        posY += 45
        # New reference image
        self.newRefBtn = QPushButton('New Reference Image', self)
        self.newRefBtn.setGeometry(posX + 30, posY, 180, 30)
        self.newRefBtn.setEnabled(False)
        self.newRefBtn.clicked.connect(self.NewRef)
        posY += 40
        # Save results checker
        self.saveResultsChk = QCheckBox('Save results', self)
        self.saveResultsChk.setGeometry(posX + 100, posY, 180, 30)
        self.saveResultsChk.setCheckable(True)
        self.saveResultsChk.setChecked(self.saveResults.value)
        self.saveResultsChk.setEnabled(True)
        self.saveResultsChk.clicked.connect(self.SaveResults)
        posY += 30
        # Rec audio checker
        self.recSoundChk = QCheckBox('Record sound', self)
        self.recSoundChk.setGeometry(posX + 100, posY, 180, 30)
        self.recSoundChk.setCheckable(True)
        self.recSoundChk.setChecked(self.recSound.value)
        self.recSoundChk.setEnabled(True)
        self.recSoundChk.clicked.connect(self.RecSound)
        posY += 30

        return posY

    def ImgMode(self):
        self.imgMode.value = self.imgModeCombo.currentText()

    def RecoboxControllerUI(self, posX, posY):
        self.paramLbl = QLabel('Recobox Controller ', self)
        self.paramLbl.setGeometry(posX, posY, 180, 30)
        self.paramLbl.setStyleSheet('font-weight: bold')
        posY += 30
        self.blinkLEDBtn = QPushButton('LED Blink', self)
        self.blinkLEDBtn.setCheckable(True)
        self.blinkLEDBtn.clicked.connect(self.ActivateLED_Blink)
        self.blinkLEDBtn.setGeometry(posX, posY, 120, 30)
        self.blinkBzBtn = QPushButton('Buzzer Blink', self)
        self.blinkBzBtn.setCheckable(True)
        self.blinkBzBtn.clicked.connect(self.RecoboxBuzz_Blink)
        self.blinkBzBtn.setGeometry(posX + 120, posY, 120, 30)
        posY += 30
        self.startLEDBtn = QPushButton('LED Start', self)
        self.startLEDBtn.setCheckable(True)
        self.startLEDBtn.clicked.connect(self.ActivateLED_On)
        self.startLEDBtn.setGeometry(posX, posY, 120, 30)
        self.startBzBtn = QPushButton('Buzzer Start', self)
        self.startBzBtn.setCheckable(True)
        self.startBzBtn.clicked.connect(self.RecoboxBuzz_On)
        self.startBzBtn.setGeometry(posX + 120, posY, 120, 30)
        posY += 30

        self.stopLEDBtn = QPushButton('LED Stop', self)
        self.stopLEDBtn.clicked.connect(self.RecoboxLED_Off)
        self.stopLEDBtn.setGeometry(posX, posY, 120, 30)
        self.stopLEDBtn.setCheckable(False)
        self.stopBzBtn = QPushButton('Buzzer Stop', self)
        self.stopBzBtn.clicked.connect(self.RecoboxBuzz_Off)
        self.stopLEDBtn.setCheckable(False)
        self.stopBzBtn.setGeometry(posX + 120, posY, 120, 30)
        posY += 30

        self.colorCombo = QComboBox(self)
        colorType = ['Red', 'Green', 'Yellow']
        self.colorCombo.addItems(colorType)
        self.colorCombo.setGeometry(posX, posY, 120, 30)
        self.colorCombo.activated[str].connect(self.RecoboxLED)

        self.rewardBtn = QPushButton('Reward', self)
        self.rewardBtn.clicked.connect(self.RecoboxReward)
        self.rewardBtn.setCheckable(False)
        self.rewardBtn.setGeometry(posX + 120, posY, 120, 30)
        posY += 50

        self.cleanBtn = QPushButton('Clean', self)
        self.cleanBtn.clicked.connect(self.RecoboxClean)
        self.cleanBtn.setGeometry(posX, posY, 120, 30)
        self.cleanBtn.setCheckable(False)

        self.refillBtn = QPushButton('Refill', self)
        self.refillBtn.clicked.connect(self.RecoboxRefill)
        self.refillBtn.setGeometry(posX + 120, posY, 120, 30)
        self.refillBtn.setCheckable(False)

        self.blinkLEDBtn.setEnabled(self.recoboxRunning.value)
        self.blinkBzBtn.setEnabled(self.recoboxRunning.value)
        self.startLEDBtn.setEnabled(self.recoboxRunning.value)
        self.startBzBtn.setEnabled(self.recoboxRunning.value)
        self.stopLEDBtn.setEnabled(self.recoboxRunning.value)
        self.stopBzBtn.setEnabled(self.recoboxRunning.value)
        self.colorCombo.setEnabled(self.recoboxRunning.value)
        self.rewardBtn.setEnabled(self.recoboxRunning.value)
        self.cleanBtn.setEnabled(self.recoboxRunning.value)
        self.refillBtn.setEnabled(self.recoboxRunning.value)

        return posY

    def StartExperiment(self):
        # Disable the monitoring when a trial is started
        self.startExpBtn.setEnabled(False)
        self.startExpBtn.setChecked(False)
        self.endExpBtn.setEnabled(True)
        self.endExpBtn.setCheckable(True)
        self.endExpBtn.setChecked(False)
        self.startTrialBtn.setEnabled(True)
        self.newRefBtn.setEnabled(True)
        self.runDLCBtn.setEnabled(False)
        self.runDetectBtn.setEnabled(False)
        self.triangulateBtn.setEnabled(False)
        self.showPos2DBtn.setEnabled(True)
        # Send command to TCP server
        self.SendCommandTCPTracking('startExperiment')
        self.log.LogText(2, 'UIController: startExperiment sent')

    def EndExperiment(self):
        self.startExpBtn.setEnabled(True)
        self.startExpBtn.setChecked(False)
        self.endExpBtn.setEnabled(False)
        self.endExpBtn.setChecked(False)
        self.newRefBtn.setEnabled(False)
        self.runDLCBtn.setEnabled(True)
        self.runDetectBtn.setEnabled(True)
        self.triangulateBtn.setEnabled(True)
        self.startTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(False)
        self.endTrialBtn.setEnabled(False)
        # Send command to TCP server
        self.SendCommandTCPTracking('endExperiment')
        self.log.LogText(2, 'UIController: endExperiment sent')

    def StartTrial(self):
        self.startTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(False)
        self.endTrialBtn.setEnabled(True)
        self.saveResultsChk.setEnabled(False)           # Disable saveResults checkbox
        self.recSoundChk.setEnabled(False)              # Disable recSound checkbox
        # Disable experiment UI fields
        self.textExpID.setEnabled(False)
        self.textSubjID.setEnabled(False)
        self.textTrialID.setEnabled(False)
        self.textCondID.setEnabled(False)
        # Send commands to TCP server
        self.SendCommandTCPTracking('expID=%s' % self.textExpID.text())
        self.SendCommandTCPTracking('subjectID=%s' % self.textSubjID.text())
        self.SendCommandTCPTracking('trialID=%s' % self.textTrialID.text())
        self.SendCommandTCPTracking('condID=%s' % self.textCondID.text())
        self.SendCommandTCPTracking('startTrial')
        self.log.LogText(2, 'UIController: startTrial sent')

    def EndTrial(self):
        self.endTrialBtn.setEnabled(False)
        self.endTrialBtn.setChecked(False)
        self.startTrialBtn.setEnabled(True)         # Enable startTrial button
        self.saveResultsChk.setEnabled(True)        # Enable saveResults checkbox
        self.recSoundChk.setEnabled(True)           # Enable recSound checkbox
        # Enable experiment UI fields
        self.textExpID.setEnabled(True)
        self.textSubjID.setEnabled(True)
        self.textTrialID.setEnabled(True)
        self.textCondID.setEnabled(True)
        # Send command to TCP server
        self.SendCommandTCPTracking('endTrial')
        self.log.LogText(2, 'UIController: endTrial sent')

    def RunDetect(self):
        if self.runDetectBtn.isChecked():
            self.imgModeTxt.setEnabled(True)
            self.imgModeCombo.setEnabled(True)
            if not self.runDLC.value:
                self.showPos2DBtn.setChecked(True)
                self.showPos2DBtn.setEnabled(True)
                self.triangulateBtn.setChecked(True)
                self.triangulateBtn.setEnabled(True)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(True)
        else:
            self.imgModeTxt.setEnabled(False)
            self.imgModeCombo.setEnabled(False)
            self.newRefBtn.setEnabled(False)
            if not self.runDLC.value:
                self.showPos2DBtn.setChecked(False)
                self.showPos2DBtn.setEnabled(False)
                self.triangulateBtn.setChecked(False)
                self.triangulateBtn.setEnabled(False)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(False)
        self.imgModeCombo.setCurrentIndex(self.imgType.index(self.imgMode.value))
        self.runDetect.value = self.runDetectBtn.isChecked()
        self.showPos2D.value = self.showPos2DBtn.isChecked()
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def RunDLC(self):
        if self.runDLCBtn.isChecked():
            self.showDLCBtn.setChecked(True)
            self.showDLCBtn.setEnabled(True)
            self.useCyclopChk.setChecked(True)
            self.useCyclopChk.setEnabled(True)
            if not self.runDetect.value:
                self.showPos2DBtn.setChecked(True)
                self.showPos2DBtn.setEnabled(True)
                self.triangulateBtn.setChecked(True)
                self.triangulateBtn.setEnabled(True)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(True)
        else:
            self.showDLCBtn.setChecked(False)
            self.showDLCBtn.setEnabled(False)
            self.useCyclopChk.setChecked(False)
            self.useCyclopChk.setEnabled(False)
            if not self.runDetect.value:
                self.showPos2DBtn.setChecked(False)
                self.showPos2DBtn.setEnabled(False)
                self.triangulateBtn.setChecked(False)
                self.triangulateBtn.setEnabled(False)
                self.sendPos3DBtn.setChecked(False)
                self.sendPos3DBtn.setEnabled(False)
        self.runDLC.value = self.runDLCBtn.isChecked()
        self.showDLC.value = self.showDLCBtn.isChecked()
        self.showPos2D.value = self.showPos2DBtn.isChecked()
        self.useCyclop.value = self.useCyclopChk.isChecked()
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def ShowPos2D(self):
        self.showPos2D.value = self.showPos2DBtn.isChecked()

    def ShowDLC(self):
        self.showDLC.value = self.showDLCBtn.isChecked()

    def UseCyclop(self):
        self.useCyclop.value = self.useCyclopChk.isChecked()

    def Triangulate(self):
        if self.triangulateBtn.isChecked():
            self.sendPos3DBtn.setChecked(True)
            self.sendPos3DBtn.setEnabled(True)
        else:
            self.sendPos3DBtn.setChecked(False)
            self.sendPos3DBtn.setEnabled(False)
        self.triangulate.value = self.triangulateBtn.isChecked()
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def SendPos3D(self):
        self.sendPos3D.value = self.sendPos3DBtn.isChecked()

    def LoadSettings(self):
        self.SendCommandTCPTracking('loadSettings')
        time.sleep(0.5)
        self.UpdateSystemSettingsUI()
        self.UpdateExperimentSettingsUI()

    def NewRef(self):
        if self.expStarted.value:
            self.SendCommandTCPTracking('newRef')
            self.log.LogText(2, 'UIController: newRef sent')
        else:
            self.log.LogText(2, 'UIController: start experiment before calling newRef')

    def SaveResults(self):
        self.saveResults.value = self.saveResultsChk.isChecked()

    def RecSound(self):
        self.recSound.value = self.recSoundChk.isChecked()

    def ResetRecoboxMonitoring(self):
        self.startLEDBtn.setChecked(False)
        self.blinkLEDBtn.setChecked(False)

    def RecoboxLED(self):
        col = str(self.colorCombo.currentText())[0]
        self.log.LogText(2, 'UIcontroller: Recobox color %s sent' % col)
        if self.blinkLEDBtn.isChecked():
            self.log.LogText(2, 'UIcontroller: Recobox LED %s BLINK sent' % col)
            self.SendCommandTCPTracking('LED %s BLINK' % col[0])
        elif self.startLEDBtn.isChecked():
            self.log.LogText(2, 'UIcontroller: Recobox LED %s ON sent' % col)
            self.SendCommandTCPTracking('LED %s ON' % col[0])

    def ActivateLED_Blink(self):
        self.blinkLEDBtn.setChecked(True)
        self.startLEDBtn.setChecked(False)
        self.RecoboxLED()

    def ActivateLED_On(self):
        self.blinkLEDBtn.setChecked(False)
        self.startLEDBtn.setChecked(True)
        self.RecoboxLED()

    def RecoboxLED_Off(self):
        self.log.LogText(2, 'UIcontroller: Recobox LED OFF sent')
        self.SendCommandTCPTracking('LED OFF')
        self.blinkLEDBtn.setChecked(False)
        self.startLEDBtn.setChecked(False)

    def RecoboxBuzz_Blink(self):
        self.log.LogText(2, 'UIcontroller: Recobox BUZZER BLINK sent')
        self.SendCommandTCPTracking('BUZZER BLINK')

    def RecoboxBuzz_On(self):
        self.log.LogText(2, 'UIcontroller: Recobox BUZZER ON sent')
        self.SendCommandTCPTracking('BUZZER ON')

    def RecoboxBuzz_Off(self):
        self.log.LogText(2, 'UIcontroller: Recobox BUZZER OFF sent')
        self.SendCommandTCPTracking('BUZZER OFF')
        self.blinkBzBtn.setChecked(False)
        self.startBzBtn.setChecked(False)

    def RecoboxReward(self):
        self.log.LogText(2, 'UIcontroller: Recobox REWARD sent')
        self.SendCommandTCPTracking('REWARD')

    def RecoboxRefill(self):   # TODO change time.sleep with a loop that block the button for the time. We need to be able to turn on/off led/buzzer even if the refill/clean button is on
        self.log.LogText(2, 'UIcontroller: Recobox REFILL sent')
        self.SendCommandTCPTracking('REFILL')
        self.rewardBtn.setEnabled(False)
        self.cleanBtn.setEnabled(False)
        time.sleep(5.2)
        self.refillBtn.setChecked(False)
        self.rewardBtn.setEnabled(True)
        self.cleanBtn.setEnabled(True)

    def RecoboxClean(self):
        self.SendCommandTCPTracking('CLEAN')
        self.log.LogText(2, 'UIcontroller: Recobox CLEAN sent')
        self.rewardBtn.setEnabled(False)
        self.refillBtn.setEnabled(False)
        time.sleep(15.6)
        self.cleanBtn.setChecked(False)
        self.rewardBtn.setEnabled(True)
        self.refillBtn.setEnabled(True)

    def closeEvent(self, event):
        self.Quit()

    def Quit(self):
        # Soft quit
        self.SendCommandTCPTracking('quit')
        self.log.LogText(2, 'UIController: quit sent ("soft" quit when TCP server is up)')

        # Hard quit (if server is down)
        self.stopRequest.value = True
        self.quit.value = True
        self.log.LogText(2, 'UIController: quit and stopRequest set to True ("hard" quit when TCP server is down)')

        QApplication.instance().quit

    def SendCommandTCPTracking(self, command, separator='\t'):
        """Send commands to TCP server running on Tracking PC"""

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as clientSocket:

            # Connects to TCP server on Tracking
            try:
                clientSocket.connect(TCPserverTracking)
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPTracking: Connection error: %s' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPTracking: Connected to TCP server')

            # Prepares binary message
            msg = str.encode(command + separator)

            # Sends command
            try:
                clientSocket.sendall(msg)  # Crashes if server is down
            except socket.error as errorMsg:
                self.log.LogText(2, 'SendCommandTCPTracking: Sending error: %s, ignoring' % errorMsg)
                return 0
            self.log.LogText(2, 'SendCommandTCPTracking: cmd=\'%s\' sent' % command)


class Log:
    """Class used to safely log the ongoing of the program (also used for debuging)"""

    def __init__(self, logLevel=int, showTime=True, __output=''):
        """Use output='' for console writing"""

        self.__lock = threading.Lock()
        self.logLevel = logLevel
        self.showTime = showTime
        if showTime:
            self.startTime = time.time_ns()
        if __output != '':
            self.__outToFile = True
            self.__stdoutCopy = sys.stdout
            sys.stdout = open(__output, 'wb')
        else:
            self.__outToFile = False

    def __del__(self):  # Called when destroying object

        del self.__lock
        if self.__outToFile:
            sys.stdout.flush()
            sys.stdout = self.__stdoutCopy

    def LogText(self, level, text):

        if self.logLevel >= level:
            self.__lock.acquire()
            if self.showTime:
                t = float(time.time_ns() - self.startTime) / 1E9
                print('%10.6f\t' % t + '  ' * (level - 1) + text)
            else:
                print('  ' * (level - 1) + text)
            self.__lock.release()


# Starts everything (if this is the main process)
if __name__ == '__main__':

    # Move to Tracking.py directory (if not already)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Multi-processing settings
    mp.set_start_method('spawn')
    mp.freeze_support()

    # Start tracking
    myTracking = Tracking()