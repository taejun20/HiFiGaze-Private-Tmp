//
//  ExperimentView.swift
//  data-collection-pretrain
//
//  Created by TJ on 6/18/25.
//

import SwiftUI
import QuartzCore

struct CalibrationView: View {
    let subject: String
    let session: String
    @Binding var calibrationCurrentTrialNum: Int
    let calibrationTotalTrialNum: Int
    @Binding var calibrationStarted: Bool
    @Binding var calibrationBackgroundImageArray: [UIImage]
    let camera: CameraController
    let dotRadius: CGFloat
    let appStartTime: Double
    @Binding var calibrationCurrentBackgroundIndex: Int

    @State private var dotPosition = CGPoint(x: -100, y: -100)
    
    @StateObject private var displayLinkCoordinatorCalibration = DisplayLinkCoordinatorCalibration()
    @Binding var calibrationFinished: Bool
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: calibrationBackgroundImageArray[calibrationCurrentBackgroundIndex])
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geometry.size.width)
                    .clipped()
                ZStack {
                    Circle()
                        .fill(Color.gray)
                        .frame(width: dotRadius * 2, height: dotRadius * 2)
                    
                    Circle()
                        .fill(Color.white)
                        .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                }
                .position(dotPosition)
            }
            .drawingGroup()
            .onAppear {
                camera.startContinuousCapture()
                setupAnimation(in: geometry.size)
                displayLinkCoordinatorCalibration.callback = { timestamp in
                    displayLinkCallback(timestamp: timestamp)
                }
                displayLinkCoordinatorCalibration.start()
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    print("✅ Calibration Trial \(calibrationCurrentTrialNum) finished.")
                    calibrationStarted = false
                    calibrationCurrentTrialNum += 1
                    if calibrationCurrentTrialNum > calibrationTotalTrialNum {
                        calibrationFinished = true
                    }
                }
            }
            .onDisappear {
                camera.stopContinuousCapture()
                displayLinkCoordinatorCalibration.stop()
            }
        }
    }
    
    func setupAnimation(in size: CGSize) {
        let locationIndex = (calibrationCurrentTrialNum - 1) % 5
        if locationIndex == 0 {
            dotPosition = CGPoint(x: size.width * 0.5, y: size.height * 0.5)
        }
        else if locationIndex == 1 {
            dotPosition = CGPoint(x: size.width * 0.1, y: size.height * 0.1)
        }
        else if locationIndex == 2 {
            dotPosition = CGPoint(x: size.width * 0.9, y: size.height * 0.1)
        }
        else if locationIndex == 3 {
            dotPosition = CGPoint(x: size.width * 0.9, y: size.height * 0.9)
        }
        else if locationIndex == 4 {
            dotPosition = CGPoint(x: size.width * 0.1, y: size.height * 0.9)
        }
        
        calibrationCurrentBackgroundIndex = (calibrationCurrentTrialNum - 1) / 5
    }
    
    // key for camera frame-screen UI synchronization
    func displayLinkCallback(timestamp: CFTimeInterval) {
        let latestVsyncStartTime = timestamp
        if calibrationStarted {
            camera.recordUIStateCalibration(
                subject: subject,
                session: session,
                trial: calibrationCurrentTrialNum,
                px_x: dotPosition.x,
                px_y: dotPosition.y,
                px_norm_x: dotPosition.x / 1000.0,
                px_norm_y: dotPosition.y / 1000.0,
                backgroundIndex: String(calibrationCurrentBackgroundIndex),    // 0: black, 1: gray, 2: white
                VSyncStartTimeOfPreviousUpdate: latestVsyncStartTime - appStartTime,
            )
        }
        
        //print ("### dotPosition: \(dotPosition), time: \(latestVsyncStartTime - appStartTime)")
        //print ("### dotPosition: \(dotPosition), backgroundA: \(backgroundImageFilenameArray[currentBackgroundIndex]), backgroundB: \(backgroundImageFilenameArray[currentBackgroundIndex + 1]), dissolve: \(dissolveProgress), displayLinkTimestamp: \(latestVsyncStartTime - appStartTime)")
    }
}

class DisplayLinkCoordinatorCalibration: ObservableObject {
    var displayLink: CADisplayLink?
    var callback: ((CFTimeInterval) -> Void)?

    func start() {
        stop()
        displayLink = CADisplayLink(target: self, selector: #selector(dlCallback))
        displayLink?.add(to: .main, forMode: .common)
    }

    func stop() {
        displayLink?.invalidate()
        displayLink = nil
    }

    @objc func dlCallback(_ link: CADisplayLink) {
        callback?(link.timestamp)
    }
}
