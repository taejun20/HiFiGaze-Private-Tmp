//
//  CameraController.swift
//  data-collection-pretrain
//
//  Created by TJ on 6/18/25.
//

import AVFoundation
import CoreImage
import UIKit

class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "cameraSessionQueue")
    private let frameSavingQueue = DispatchQueue(label: "frameSavingQueue", qos: .background)
    private let colorSpace = CGColorSpaceCreateDeviceRGB()
    private let ciContext = CIContext()

    var continuousCaptureOn: Bool = false
    var captureInterval: TimeInterval = 1.0 / 30.0 // 30 FPS default
    private var lastCaptureTime: CFTimeInterval = 0
    private var frameCounter: Int = 0
    var appStartTime: Double = -1
    
    private var framesDir: URL!
    private var subject: String = ""
    private var sessionID: String = ""

    private var uiStateLogs: [UIStateLog] = []
    private let uiStateLogsLock = NSLock()
    private var captureDevice: AVCaptureDevice?
    
    override init() {
        super.init()
        setupSession()
    }
    
    func startContinuousCapture() {
        continuousCaptureOn = true
        lastCaptureTime = CACurrentMediaTime()
    }
    
    func stopContinuousCapture() {
        continuousCaptureOn = false
    }
    
    func startSession() {
        sessionQueue.async {
            if !self.session.isRunning {
                self.session.startRunning()
            }
        }
    }
    
    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }
    
    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .inputPriority
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            print("❌ Could not find front camera")
            return
        }
        self.captureDevice = device

        // 4K format that supports 60 FPS
        var selectedFormat: AVCaptureDevice.Format? = nil
        for format in device.formats {
            let desc = format.formatDescription
            let dims = CMVideoFormatDescriptionGetDimensions(desc)
            if dims.width == 3840 && dims.height == 2160 {
                for range in format.videoSupportedFrameRateRanges {
                    if range.maxFrameRate >= 60 {
                        selectedFormat = format
                        break
                    }
                }
            }
            if selectedFormat != nil { break }
        }
        guard let format = selectedFormat else {
            print("❌ No 4K@60fps format found")
            return
        }

        // Lock and configure device
        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
            device.unlockForConfiguration()
        } catch {
            print("❌ Failed to configure camera: \(error)")
            return
        }

        // Create and add input
        guard let input = try? AVCaptureDeviceInput(device: device),
              self.session.canAddInput(input) else {
            print("❌ Could not create or add input")
            return
        }
        self.session.addInput(input)
        print("🔍 Active format set: \(device.activeFormat)")
        print("🔍 Min Max Frame duration: \(device.activeVideoMinFrameDuration), Max duration: \(device.activeVideoMaxFrameDuration)")

        // Add video output with high priority queue
        if session.canAddOutput(videoOutput) {
            videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoOutputQueue", qos: .userInteractive))
            session.addOutput(videoOutput)
        }
        session.commitConfiguration()
    }

    func configure(subject: String, session: String) {
        self.subject = subject
        self.sessionID = session
        self.framesDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("frames")
            .appendingPathComponent("p\(subject)")
            .appendingPathComponent("\(sessionID)")
        try? FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)
    }
        
    func clearTrialData(deletingTrialNum: Int) {
        guard let framesDir = framesDir else { return }
        
        // 1. delete saved camera frames
        uiStateLogsLock.lock()
        let frameNumbersForDeletingTrial = uiStateLogs      // Get frame numbers associated with this trial
            .filter { $0.trial >= deletingTrialNum && $0.frameNum != "-" }
            .compactMap { Int($0.frameNum) }
        var deletedCount = 0
        for frameNumber in frameNumbersForDeletingTrial {
            let frameFileName = String(format: "%05d.jpg", frameNumber)
            let frameFileURL = framesDir.appendingPathComponent(frameFileName)
            if FileManager.default.fileExists(atPath: frameFileURL.path) {
                try? FileManager.default.removeItem(at: frameFileURL)
                deletedCount += 1
            }
        }
        
        // 2. correct frameCounter, looking at the latest frame till the redoing trial
        let previousTrialNum = deletingTrialNum - 1
        let frameCountTillPreviousTrial: Int
        if previousTrialNum > 0 {
            frameCountTillPreviousTrial = uiStateLogs
                .filter { $0.trial == previousTrialNum && $0.frameNum != "-" }
                .compactMap { Int($0.frameNum) }
                .max() ?? 0
        } else {
            frameCountTillPreviousTrial = 0  // If redoing the first trial, start from 0
        }
        frameCounter = frameCountTillPreviousTrial

        // 3. delete the saved rows in the uiStateLogs array
        uiStateLogs.removeAll { $0.trial >= deletingTrialNum }
        uiStateLogsLock.unlock()
    }
    
    func saveUIStateLogs() {
        uiStateLogsLock.lock()
        let csvHeader = "subject,session,trial,px_x,px_y,px_norm_x,px_norm_y,VSyncStartTimeOfPreviousUpdate,frameNum,frameEstimatedReadoutStartTime\n"
        let csvRows = uiStateLogs.map { log in
            "\(log.subject),\(log.session),\(log.trial),\(log.px_x),\(log.px_y),\(log.px_norm_x),\(log.px_norm_y),\(log.VSyncStartTimeOfPreviousUpdate),\(log.frameNum),\(log.frameEstimatedReadoutStartTime)"
        }
        let csvString = csvHeader + csvRows.joined(separator: "\n")
        uiStateLogsLock.unlock()
        
        let csvURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("frames")
            .appendingPathComponent("p\(subject)")
            .appendingPathComponent("p\(subject)_\(sessionID)_data.csv")
        
        do {
            try csvString.write(to: csvURL, atomically: true, encoding: .utf8)
            print("✅ UI state logs saved to \(csvURL)")
        } catch {
            print("❌ Failed to write UI state logs: \(error)")
        }
    }
    
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let currentTime = CACurrentMediaTime()
        let shouldCapture = continuousCaptureOn && (currentTime - lastCaptureTime) >= captureInterval
        if shouldCapture {
            lastCaptureTime = currentTime
            frameCounter += 1
            
            let presentationTimestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer)) - appStartTime
            guard let exposureTime = captureDevice?.exposureDuration else {
                return
            }
            let exposureTimestamp = CMTimeGetSeconds(exposureTime)
            let estimatedReadoutStartTime = presentationTimestamp + exposureTimestamp
            //print("frameCounter: \(frameCounter), pts: \(presentationTimestamp), exposure: \(exposureTimestamp), estimatedReadoutStartTime: \(estimatedReadoutStartTime)")
            
            uiStateLogsLock.lock()
            let startIndex = max(0, uiStateLogs.count - 5)
            let endIndex = uiStateLogs.count
            var closestIndex = -1
            var closestTimeDiff = Double.infinity
            for i in startIndex..<endIndex {
                let uiStateLog = uiStateLogs[i]
                let timeDiff = estimatedReadoutStartTime - uiStateLog.VSyncStartTimeOfPreviousUpdate
                
                // make sure that the timediff is greater than 15ms (0.015s)
                if timeDiff > 0 && timeDiff < closestTimeDiff {
                    closestTimeDiff = timeDiff
                    closestIndex = i
                }
            }
            if closestIndex > 0 {
                if estimatedReadoutStartTime - uiStateLogs[closestIndex - 1].VSyncStartTimeOfPreviousUpdate > 0.1 {
                    uiStateLogs[closestIndex].frameNum = String(format: "%05d", frameCounter)
                    uiStateLogs[closestIndex].frameEstimatedReadoutStartTime = estimatedReadoutStartTime
                }
                else {
                    uiStateLogs[closestIndex - 1].frameNum = String(format: "%05d", frameCounter)
                    uiStateLogs[closestIndex - 1].frameEstimatedReadoutStartTime = estimatedReadoutStartTime
                }
            }
            uiStateLogsLock.unlock()
            
            guard let pxbuf = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let ciImage = CIImage(cvImageBuffer: pxbuf)
            let jpgData = ciContext.jpegRepresentation(of: ciImage, colorSpace: self.colorSpace)
            let frameName = String(format: "%05d", frameCounter)
            frameSavingQueue.async { [weak self] in
                guard let self = self,
                      let jpgData = jpgData,
                      let framesDir = self.framesDir else { return }
                autoreleasepool {
                    let url = framesDir.appendingPathComponent("\(frameName).jpg")
                    do {
                        try jpgData.write(to: url, options: [.atomic])
                    } catch {
                        print("❌ JPEG write failed:", error)
                    }
                }
            }
        }
    }
    
    func recordUIState(subject: String, session: String, trial: Int, px_x: CGFloat, px_y: CGFloat, px_norm_x: CGFloat, px_norm_y: CGFloat, VSyncStartTimeOfPreviousUpdate: CFTimeInterval) {
        let uiStateLog = UIStateLog(
            subject: subject,
            session: session,
            trial: trial,
            px_x: px_x,
            px_y: px_y,
            px_norm_x: px_norm_x,
            px_norm_y: px_norm_y,
            VSyncStartTimeOfPreviousUpdate: VSyncStartTimeOfPreviousUpdate,
            frameNum: "-",  // Initially unassigned
            frameEstimatedReadoutStartTime: -1
        )
        uiStateLogsLock.lock()
        uiStateLogs.append(uiStateLog)
        uiStateLogsLock.unlock()
    }
}

struct UIStateLog: Codable {
    let subject: String
    let session: String
    let trial: Int
    let px_x: CGFloat
    let px_y: CGFloat
    let px_norm_x: CGFloat
    let px_norm_y: CGFloat
    let VSyncStartTimeOfPreviousUpdate: CFTimeInterval
    var frameNum: String  // Will be "-" initially, then updated with frame number
    var frameEstimatedReadoutStartTime: CFTimeInterval
}
