//
//  ExperimentView.swift
//  data-collection-pretrain
//
//  Created by TJ on 6/18/25.
//

import SwiftUI
import QuartzCore

struct ExperimentView: View {
    let subject: String
    let session: String
    let totalTrialNum: Int
    @Binding var trialStarted: Bool
    @Binding var currentTrialNum: Int
    @Binding var trialTestOrderArray: [String]
    @Binding var backgroundImageArray: [UIImage]
    @Binding var backgroundImageFilenameArray: [String]
    @Binding var currentBackgroundIndex: Int
    @Binding var nextBackgroundIndex: Int
    @State private var phase: Int = 0 // 0: static, 1: dissolve
    @State private var phaseStartTime: CFTimeInterval = 0
    @State private var dissolveProgress: CGFloat = 0
    let staticDuration: CFTimeInterval = 0.2
    let dissolveDuration: CFTimeInterval = 0.2

    let camera: CameraController
    let dotRadius: CGFloat
    let horizontalLaneCount: Int
    let verticalLaneCount: Int
    let appStartTime: Double
    
    @State private var dotPosition = CGPoint(x: -100, y: -100)
    
    @State private var trialStartTime: CFTimeInterval = 0
    @State private var dotStartPoint = CGPoint.zero
    @State private var dotEndPoint = CGPoint.zero
    @State private var isDotMoving = false
    @State private var isHorizontal = false
    @State private var isReversed = false
    let stepSize: Double = 100.0

    @StateObject private var displayLinkCoordinator = DisplayLinkCoordinator()

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: backgroundImageArray[currentBackgroundIndex])
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: geometry.size.width)
                    .clipped()
                    .opacity(phase == 0 ? 1.0 : 1.0 - dissolveProgress)
                
                if phase == 1 {
                    Image(uiImage: backgroundImageArray[nextBackgroundIndex])
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: geometry.size.width)
                                .clipped()
                                .opacity(dissolveProgress)
                }
                
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
                phaseStartTime = CACurrentMediaTime()
                displayLinkCoordinator.callback = { timestamp in
                    displayLinkCallback(timestamp: timestamp)
                }
                displayLinkCoordinator.start()
            }
            .onDisappear {
                camera.stopContinuousCapture()
                displayLinkCoordinator.stop()
            }
        }
    }
   
    func setupAnimation(in size: CGSize) {
        let trialTypeStr = trialTestOrderArray[currentTrialNum-1]
        isHorizontal = trialTypeStr.hasPrefix("h")
        isReversed = trialTypeStr.hasSuffix("r")
        let laneIndexStr = trialTypeStr.dropFirst().dropLast(isReversed ? 1 : 0)
        let laneIndex = CGFloat(Int(laneIndexStr) ?? 0)
        let leftX = size.width * 0.1
        let topY = size.height * 0.1
        let laneHeight = (size.height * 0.8) / CGFloat(horizontalLaneCount - 1)
        let laneWidth = (size.width * 0.8) / CGFloat(verticalLaneCount - 1)
        if isHorizontal {
            let y = topY + laneHeight * laneIndex
            dotStartPoint = CGPoint(x: isReversed ? leftX + 100 : leftX, y: y)
            dotEndPoint = CGPoint(x: isReversed ? leftX : size.width * 0.9, y: y)
        } else {
            let x = leftX + laneWidth * laneIndex
            dotStartPoint = CGPoint(x: x, y: isReversed ? topY + 100 : topY)
            dotEndPoint = CGPoint(x: x, y: isReversed ? topY : size.height * 0.9)
        }
        dotPosition = dotStartPoint
        trialStartTime = CACurrentMediaTime()
        isDotMoving = true
    }
    
    // key for camera frame-screen UI synchronization
    func displayLinkCallback(timestamp: CFTimeInterval) {
        let latestVsyncStartTime = timestamp
        if isDotMoving {
            updateDotPosition(at: latestVsyncStartTime)
        }
        updateBackground(at: latestVsyncStartTime)
        if trialStarted {
            camera.recordUIState(
                subject: subject,
                session: session,
                trial: currentTrialNum,
                px_x: dotPosition.x,
                px_y: dotPosition.y,
                px_norm_x: dotPosition.x / 1000.0,
                px_norm_y: dotPosition.y / 1000.0,
                backgroundA: backgroundImageFilenameArray[currentBackgroundIndex],
                backgroundB: backgroundImageFilenameArray[currentBackgroundIndex+1],
                dissolve: dissolveProgress,
                VSyncStartTimeOfPreviousUpdate: latestVsyncStartTime - appStartTime,
            )
        }
        
        //print ("### dotPosition: \(dotPosition), time: \(latestVsyncStartTime - appStartTime)")
        //print ("### dotPosition: \(dotPosition), backgroundA: \(backgroundImageFilenameArray[currentBackgroundIndex]), backgroundB: \(backgroundImageFilenameArray[currentBackgroundIndex + 1]), dissolve: \(dissolveProgress), displayLinkTimestamp: \(latestVsyncStartTime - appStartTime)")
    }

    func updateDotPosition(at currentTime: CFTimeInterval) {
        let elapsedTime = currentTime - trialStartTime
        let totalDistance = isHorizontal ?
        abs(dotEndPoint.x - dotStartPoint.x) :
        abs(dotEndPoint.y - dotStartPoint.y)
    
        let progress = min(elapsedTime * stepSize / totalDistance, 1.0)
        if progress >= 1.0 {
            isDotMoving = false
            dotPosition = dotEndPoint
            if currentTrialNum <= totalTrialNum {
                print("✅ Trial \(currentTrialNum) finished.")
                trialStarted = false
                currentTrialNum += 1
            }
        } else {
            if isHorizontal {
                let newX = isReversed ?
                    dotStartPoint.x - (totalDistance * progress) :
                    dotStartPoint.x + (totalDistance * progress)
                dotPosition = CGPoint(x: newX, y: dotStartPoint.y)
            } else {
                let newY = isReversed ?
                    dotStartPoint.y - (totalDistance * progress) :
                    dotStartPoint.y + (totalDistance * progress)
                dotPosition = CGPoint(x: dotStartPoint.x, y: newY)
            }
        }
    }
    
    func updateBackground(at latestVsyncStartTime: CFTimeInterval) {
        let elapsed = latestVsyncStartTime - phaseStartTime
        if phase == 0 { // static phase
            if elapsed >= staticDuration {
                phase = 1
                phaseStartTime = latestVsyncStartTime
                dissolveProgress = 0
                nextBackgroundIndex = currentBackgroundIndex + 1
            }
        }
        else if phase == 1 {
            let progress = min(CGFloat(elapsed / dissolveDuration), 1.0)
            dissolveProgress = progress
            if progress >= 1.0 {
                currentBackgroundIndex = nextBackgroundIndex
                phase = 0
                phaseStartTime = latestVsyncStartTime
                dissolveProgress = 0
            }
        }
    }
}

class DisplayLinkCoordinator: ObservableObject {
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
