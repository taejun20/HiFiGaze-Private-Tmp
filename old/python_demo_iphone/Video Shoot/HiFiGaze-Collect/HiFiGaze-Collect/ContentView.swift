//
//  ContentView.swift
//  data-collection-pretrain
//
//  Created by TJ on 6/18/25.
//

import SwiftUI

struct ContentView: View {
    @State private var subject: String = ""
    @State private var session: String = ""
    
    let totalTrialNum = 28
    @State private var subjectSelected = false
    @State private var sessionSelected = false
    @State private var trialStarted = false
    @State private var currentTrialNum = 1  // 1-based indexing
    let horizontalLaneCount = 18
    let verticalLaneCount = 10
    let dotRadius: CGFloat = 10
    @State var trialTestOrderArray: [String] = []
    
    @State private var appStartTime: Double = CACurrentMediaTime()
    @State private var expFinished = false
    @StateObject private var camera = CameraController()
    private let countdownTimer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    @State private var countdown: Double = 5.0

    var body: some View {
        ZStack {
            if subjectSelected {
                if sessionSelected {
                    if expFinished {
                        VStack(spacing: 40) {
                            Text("p\(subject), \(session)")
                                .font(.title)
                            
                            Text(String(format: "%.1f s", countdown))          // ⇦ shows “5.0 s … 0.0 s”
                                .font(.title)
                                .monospacedDigit()
                                .onReceive(countdownTimer) { _ in
                                    if countdown > 0 {                       // tick every 0.1 s
                                        countdown = max(countdown - 0.1, 0)
                                    } else {
                                        countdownTimer.upstream.connect().cancel() // stop when done
                                    }
                                }

                            
                            Text("✅ Experiment Finished")
                                .font(.largeTitle)
                                .foregroundColor(.green)
                        }
                    }
                    else {
                        if currentTrialNum > totalTrialNum {
                            ZStack {
                                VStack (spacing: 20) {
//                                        Button("Redo Last Trial") {
//                                            redoLastTrial()
//                                        }
//                                        .font(.largeTitle)
//                                        .padding()
//                                        .background(Color.white)
//                                        .cornerRadius(10)
                                    
                                    Text("p\(subject), \(session)")
                                        .font(.title)
                                    
                                    Text("All Trial Done")
                                        .font(.largeTitle)

                                    Button ("Complete") {
                                        expFinished = true
                                        saveCSV()
                                    }
                                    .font(.largeTitle)
                                    .padding()
                                    .background(Color.white)
                                    .cornerRadius(10)
                                }
                            }
                        } else if trialStarted {
                            ExperimentView(
                                subject: subject,
                                session: session,
                                totalTrialNum: totalTrialNum,
                                trialStarted: $trialStarted,
                                currentTrialNum: $currentTrialNum,
                                trialTestOrderArray: $trialTestOrderArray,
                                camera: camera,
                                dotRadius: dotRadius,
                                horizontalLaneCount: horizontalLaneCount,
                                verticalLaneCount: verticalLaneCount,
                                appStartTime: appStartTime
                            )
                        } else {
                            ZStack {
                                VStack (spacing: 20) {
//                                        if currentTrialNum > 1 {
//                                            Button("Redo Last Trial") {
//                                                redoLastTrial()
//                                            }
//                                            .font(.largeTitle)
//                                            .padding(.bottom, 10)
//                                        }
                                    
                                    Text("p\(subject), session: \(session)")
                                        .font(.title)
                                    
                                    Text("Trial \(min(currentTrialNum, totalTrialNum)) / \(totalTrialNum)")
                                        .font(.title)

                                    Button("Start") {
                                        trialStarted = true
                                    }
                                    .font(.largeTitle)
                                    .padding()
                                }
                                Circle()
                                    .fill(Color.gray)
                                    .frame(width: dotRadius * 2, height: dotRadius * 2)
                                    .position(getInitialDotPosition(trial: currentTrialNum))
                                
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                    .position(getInitialDotPosition(trial: currentTrialNum))
                            }
                        }
                    }
                }
                else
                {
                    VStack(spacing: 45) {
                        HStack(spacing: 60) {
                            Button("s1-1") {
                                if !sessionSelected {   // handling quick double tap case
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "1p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s1-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "1s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s2-1") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "2p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s2-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "2s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s3-1") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "3p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s3-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "3s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s4-1") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "4p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s4-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "4s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s5-1") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "5p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s5-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "5s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s6-1") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)" } + (0...verticalLaneCount-1).map { "v\($0)" }
                                    session = "6p"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("s6-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...horizontalLaneCount-1).map { "h\($0)r" } + (0...verticalLaneCount-1).map { "v\($0)r" }
                                    session = "6s"
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                    }
                }
            }
            else
            {
                VStack(spacing: 60) {
                    HStack(spacing: 60) {
                        Button("p99") {
                            subjectSelected = true
                            subject = "99"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                }
            }
        }
        .onAppear() {
            camera.appStartTime = appStartTime
            camera.startSession()
        }
        .onDisappear {
            camera.stopSession()
        }
    }
    
    func getInitialDotPosition(trial: Int) -> CGPoint {
        let trialTypeStr = trialTestOrderArray[trial-1]
        let isHorizontal = trialTypeStr.hasPrefix("h")
        let isReversed = trialTypeStr.hasSuffix("r")
        let laneIndexStr = trialTypeStr.dropFirst().dropLast(isReversed ? 1 : 0)
        let laneIndex = CGFloat(Int(laneIndexStr) ?? 0)
   
        let width = 430.0
        let height = 839.0
        let leftX = width * 0.1
        let topY = height * 0.1
        let laneHeight = (height * 0.8) / CGFloat(horizontalLaneCount - 1)
        let laneWidth = (width * 0.8) / CGFloat(verticalLaneCount - 1)
        
        var dotStartPoint: CGPoint = .zero
        if isHorizontal {
            let y = topY + laneHeight * laneIndex
            dotStartPoint = CGPoint(x: isReversed ? leftX + 100 : leftX, y: y)
        } else {
            let x = leftX + laneWidth * laneIndex
            dotStartPoint = CGPoint(x: x, y: isReversed ? topY + 100 : topY)
        }
            
        return dotStartPoint
    }
    
    func redoLastTrial() {
        currentTrialNum -= 1
        camera.clearTrialData(deletingTrialNum: currentTrialNum)
        trialStarted = true
    }
    
    func saveCSV() {
        camera.saveUIStateLogs()
    }
}
