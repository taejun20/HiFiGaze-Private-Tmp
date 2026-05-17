//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import SwiftUI

struct ContentView: View {
    @State private var subject = ""
    @State private var session = ""
    
    let totalTrialNum = 45
    @State private var subjectSelected = false
    @State private var sessionSelected = false
    @State private var focusSelected = false
    @ObservedObject var trialState = TrialState.shared
    @ObservedObject var logger = TrialLogger.shared

    let horizontalLaneCount = 9
    let verticalLaneCount = 5
    let dotRadius: CGFloat = 10
    @State var trialTestOrderArray: [Int] = Array(0...44)
    let trialDuration = 1.0     // seconds
        
    @State private var expFinished = false
    @State private var backgroundImageArray: [UIImage] = [] // Load all images into memory
    @State private var backgroundImageFilenameArray: [String] = []
    
    @State private var isFetching = false
    @State private var statusMessage = ""
    @ObservedObject private var trueDepth = TrueDepthManager.shared

    @State private var isCalibration = false
    @State private var calibrationStarted = false
    @State private var calibrationFinished = false
    @State private var calibrationCurrentTrialNum = 1   // 1-based indexing
    @State private var calibrationTotalTrialNum = 15
    @State private var calibrationBackgroundImageArray: [UIImage] = []  // Load all images into memory once
    @State private var calibrationBackgroundImageFilenameArray: [String] = []
    @State private var calibrationCurrentBackgroundIndex = 0

    var body: some View {
        ZStack {
            if subjectSelected {
                if sessionSelected {
                    if focusSelected {
                        if !isCalibration {
                            if trialState.trialReviewCompleted {
                                if expFinished {
                                    VStack(spacing: 40) {
                                        Text("p\(subject), session: \(session)")
                                            .font(.title)
                                        
                                        Text("✅ Experiment Finished")
                                            .font(.largeTitle)
                                            .foregroundColor(.green)
                                    }
                                }
                                else {
                                    if trialState.currentTrialNum > totalTrialNum {
                                        ZStack {
                                            VStack (spacing: 20) {
                                                Button("Set Focus") {
                                                    focusSelected = false
                                                }
                                                .font(.largeTitle)
                                                .padding()
                                                
                                                Text("p\(subject), session: \(session)")
                                                    .font(.title)
                                                
                                                Text("All Trial Done")
                                                    .font(.largeTitle)
                                                
                                                Button ("Complete Complete") {
                                                    logger.exportCSV(subject: subject, session: session)
                                                    expFinished = true
                                                }
                                                .frame(width: 400)
                                                .font(.largeTitle)
                                                .padding()
                                                .background(Color.white)
                                                .cornerRadius(10)
                                            }
                                            Circle()
                                                .fill(Color.gray)
                                                .frame(width: dotRadius * 2, height: dotRadius * 2)
                                                .position(getDotPosition(trial: trialState.currentTrialNum - 1))
                                            
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                                .position(getDotPosition(trial: trialState.currentTrialNum - 1))
                                        }
                                    } else if trialState.trialStarted {
                                        ExperimentView(
                                            subject: subject,
                                            session: session,
                                            totalTrialNum: totalTrialNum,
                                            trialTestOrderArray: $trialTestOrderArray,
                                            backgroundImageArray: $backgroundImageArray,
                                            backgroundImageFilenameArray: $backgroundImageFilenameArray,
                                            dotRadius: dotRadius,
                                            horizontalLaneCount: horizontalLaneCount,
                                            verticalLaneCount: verticalLaneCount,
                                            trialDuration: trialDuration
                                        )
                                    } else {
                                        ZStack {
                                            VStack (spacing: 20) {
                                                Button("Set Focus") {
                                                    focusSelected = false
                                                }
                                                .font(.largeTitle)
                                                .padding()
                                                
                                                
                                                Text("p\(subject), session: \(session)")
                                                    .font(.title)
                                                
                                                Text("Trial \(trialState.currentTrialNum) / \(totalTrialNum)")
                                                    .font(.title)
                                                
                                                Button("Start Start Start Start Start") {
                                                    let dotPos = getDotPosition(trial: trialState.currentTrialNum)
                                                    let bgName = backgroundImageFilenameArray[trialState.currentBackgroundIndex]
                                                    logger.logTrial(subject: subject, session: session, trial: trialState.currentTrialNum, position: dotPos, background: bgName)
                                                    
                                                    trialState.trialStarted = true
                                                    TCPClient.shared.send(message: "start\n")
                                                }
                                                .frame(width: 400)
                                                .font(.largeTitle)
                                                .padding()
                                                .background(Color.white)
                                                .cornerRadius(10)
                                            }
                                            Circle()
                                                .fill(Color.gray)
                                                .frame(width: dotRadius * 2, height: dotRadius * 2)
                                                .position(getDotPosition(trial: trialState.currentTrialNum))
                                            
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                                .position(getDotPosition(trial: trialState.currentTrialNum))
                                        }
                                    }
                                }
                            }
                            else {
                                TrialReviewView(focusSelected: $focusSelected, subject: subject, session: session)
                            }
                        }
                        else {
                            if trialState.trialReviewCompleted {
                                if calibrationFinished {
                                    VStack(spacing: 40) {
                                        Text("p\(subject), session: \(session)")
                                            .font(.title)
                                        
                                        Text("✅ Experiment Finished")
                                            .font(.largeTitle)
                                            .foregroundColor(.green)
                                    }
                                }
                                else {
                                    if trialState.currentTrialNum > calibrationTotalTrialNum {
                                        ZStack {
                                            VStack (spacing: 20) {
                                                Button("Set Focus") {
                                                    focusSelected = false
                                                }
                                                .font(.largeTitle)
                                                .padding()
                                                
                                                Text("p\(subject), session: \(session)")
                                                    .font(.title)
                                                
                                                Text("All Trial Done")
                                                    .font(.largeTitle)
                                                
                                                Button ("Complete") {
                                                    logger.exportCSV(subject: subject, session: session)
                                                    calibrationFinished = true
                                                }
                                                .font(.largeTitle)
                                                .padding()
                                                .background(Color.white)
                                                .cornerRadius(10)
                                            }
                                            Circle()
                                                .fill(Color.gray)
                                                .frame(width: dotRadius * 2, height: dotRadius * 2)
                                                .position(getDotPosition(trial: trialState.currentTrialNum - 1))
                                            
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                                .position(getDotPosition(trial: trialState.currentTrialNum - 1))
                                            
                                        }
                                    } else if trialState.trialStarted {
                                        CalibrationView(
                                            subject: subject,
                                            session: session,
                                            calibrationTotalTrialNum: calibrationTotalTrialNum,
                                            calibrationBackgroundImageArray: $calibrationBackgroundImageArray,
                                            dotRadius: dotRadius,
                                            horizontalLaneCount: horizontalLaneCount,
                                            verticalLaneCount: verticalLaneCount,
                                            trialDuration: trialDuration
                                        )
                                    } else {
                                        ZStack {
                                            VStack (spacing: 20) {
                                                Button("Set Focus") {
                                                    focusSelected = false
                                                }
                                                .font(.largeTitle)
                                                .padding()
                                                
                                                
                                                Text("p\(subject), session: \(session)")
                                                    .font(.title)
                                                
                                                Text("Trial \(trialState.currentTrialNum) / \(calibrationTotalTrialNum)")
                                                    .font(.title)
                                                
                                                Button("Start Start Start Start Start") {
                                                    let dotPos = getDotPosition(trial: trialState.currentTrialNum)
                                                    let bgName = calibrationBackgroundImageFilenameArray[trialState.calibrationCurrentBackgroundIndex]
                                                    logger.logTrial(subject: subject, session: session, trial: trialState.currentTrialNum, position: dotPos, background: bgName)

                                                    trialState.trialStarted = true
                                                    TCPClient.shared.send(message: "start\n")
                                                }
                                                .frame(width: 400)
                                                .font(.largeTitle)
                                                .padding()
                                                .background(Color.white)
                                                .cornerRadius(10)
                                            }
                                            Circle()
                                                .fill(Color.gray)
                                                .frame(width: dotRadius * 2, height: dotRadius * 2)
                                                .position(getDotPosition(trial: trialState.currentTrialNum))
                                            
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                                .position(getDotPosition(trial: trialState.currentTrialNum))
                                        }
                                    }
                                }
                            }
                            else {
                                TrialReviewView(focusSelected: $focusSelected, subject: subject, session: session)
                            }
                        }
                    }
                    else {
                        FocusSetView(focusSelected: $focusSelected)
                    }                    
                } else {
                    VStack(spacing: 45) {
                        Button("practice") {
                            if !sessionSelected {   // handling quick double tap case
                                sessionSelected = true
                                logger.startSession()
                                trialTestOrderArray = trialTestOrderArray.shuffled()
                                session = "0"
                                loadBackgroundDataset()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 160)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        
                        Button("session 1") {
                            if !sessionSelected {   // handling quick double tap case
                                sessionSelected = true
                                logger.startSession()
                                trialTestOrderArray = trialTestOrderArray.shuffled()
                                session = "1"
                                loadBackgroundDataset()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                                                
                        Button("session 2") {
                            if !sessionSelected {
                                sessionSelected = true
                                logger.startSession()
                                trialTestOrderArray = trialTestOrderArray.shuffled()
                                session = "2"
                                loadBackgroundDataset()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        
                        Button("session 3") {
                            if !sessionSelected {
                                sessionSelected = true
                                logger.startSession()
                                trialTestOrderArray = trialTestOrderArray.shuffled()
                                session = "3"
                                loadBackgroundDataset()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        
                        Button("session 4") {
                            if !sessionSelected {
                                sessionSelected = true
                                logger.startSession()
                                trialTestOrderArray = trialTestOrderArray.shuffled()
                                session = "4"
                                loadBackgroundDataset()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        
                        Button("calsit") {
                            if !sessionSelected {
                                sessionSelected = true
                                isCalibration = true
                                logger.startSession()
                                session = "calsit"
                                loadBackgroundDatasetCalibration()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        }
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        
                        Button("calstand") {
                            if !sessionSelected {
                                sessionSelected = true
                                isCalibration = true
                                logger.startSession()
                                session = "calstand"
                                loadBackgroundDatasetCalibration()
                                TCPClient.shared.subject = subject
                                TCPClient.shared.session = session
                            }
                        } // Good Luck ;)
                        .frame(width: 200)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                }
            }
            else {
                VStack(spacing: 60) {
                    HStack(spacing: 60) {
                        Button("p1") {
                            subjectSelected = true
                            subject = "1"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        Button("p2") {
                            subjectSelected = true
                            subject = "2"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                    HStack(spacing: 60) {
                        Button("p3") {
                            subjectSelected = true
                            subject = "3"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        Button("p4") {
                            subjectSelected = true
                            subject = "4"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                    HStack(spacing: 60) {
                        Button("p5") {
                            subjectSelected = true
                            subject = "5"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        Button("p6") {
                            subjectSelected = true
                            subject = "6"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                    HStack(spacing: 60) {
                        Button("p7") {
                            subjectSelected = true
                            subject = "7"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        Button("p8") {
                            subjectSelected = true
                            subject = "8"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                    }
                    HStack(spacing: 60) {
                        Button("p9") {
                            subjectSelected = true
                            subject = "9"
                        }
                        .frame(width: 100)
                        .font(.largeTitle)
                        .background(Color.white)
                        .cornerRadius(10)
                        Button("p10") {
                            subjectSelected = true
                            subject = "10"
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
            TCPClient.shared.connect { _ in }
        }
    }
    
    func getDotPosition(trial: Int) -> CGPoint {
        print("trial: \(trial)")
        if !isCalibration {
            let positionIndex = trialTestOrderArray[trial-1]
            let width = 430.0
            let height = 839.0
            let leftX = width * 0.1
            let topY = height * 0.1

            let horizontalInterval = width * 0.8 / CGFloat(verticalLaneCount - 1)
            let verticalInterval = height * 0.8 / CGFloat(horizontalLaneCount - 1)

            // Compute grid coordinates
            let col = positionIndex % verticalLaneCount
            let row = positionIndex / verticalLaneCount

            let x = leftX + CGFloat(col) * horizontalInterval
            let y = topY + CGFloat(row) * verticalInterval
            
            return CGPoint(x: x, y: y)
        }
        else {
            var ret = CGPoint.zero
            let width = 430.0
            let height = 839.0
            let locationIndex = (trial - 1) % 5

            if locationIndex == 0 {
                ret = CGPoint(x: width * 0.5, y: height * 0.5)
            }
            else if locationIndex == 1 {
                ret = CGPoint(x: width * 0.1, y: height * 0.1)
            }
            else if locationIndex == 2 {
                ret = CGPoint(x: width * 0.9, y: height * 0.1)
            }
            else if locationIndex == 3 {
                ret = CGPoint(x: width * 0.9, y: height * 0.9)
            }
            else if locationIndex == 4 {
                ret = CGPoint(x: width * 0.1, y: height * 0.9)
            }
            return ret
        }
    }

    func loadBackgroundDataset() {
        backgroundImageFilenameArray.removeAll()
        backgroundImageArray.removeAll()
        let screenshotDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("backgrounds-study2/p\(subject)/s\(session)")
        
        print("📂 screenshotDir: \(screenshotDir.path)")

        do {
            let files = try FileManager.default.contentsOfDirectory(at: screenshotDir, includingPropertiesForKeys: nil)
            backgroundImageFilenameArray = files
                .filter { $0.pathExtension == "jpg" }
                .map { $0.deletingPathExtension().lastPathComponent }
                .shuffled()
            
            for filename in backgroundImageFilenameArray {
                let fileURL = screenshotDir.appendingPathComponent("\(filename).jpg")
                if let data = try? Data(contentsOf: fileURL),
                   let uiImage = UIImage(data: data) {
                    backgroundImageArray.append(uiImage)
                } else {
                    print("❌ Failed to load background image: \(filename)")
                }
            }
            print("✅ backgroundImageArray Load Complete, count: \(backgroundImageArray.count)")
        } catch {
            print("❌ Error loading background dataset from local storage: \(error)")
        }
    }
    
    func loadBackgroundDatasetCalibration() {
        calibrationBackgroundImageArray.removeAll()
        let screenshotDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("backgrounds_calibration")
        
        do {
            let files = try FileManager.default.contentsOfDirectory(at: screenshotDir, includingPropertiesForKeys: nil)
            calibrationBackgroundImageFilenameArray = files
                .filter { $0.pathExtension == "jpg" }
                .map { $0.deletingPathExtension().lastPathComponent }
                .sorted { Int($0)! < Int($1)! }

            for filename in calibrationBackgroundImageFilenameArray {
                let fileURL = screenshotDir.appendingPathComponent("\(filename).jpg")
                if let data = try? Data(contentsOf: fileURL),
                   let uiImage = UIImage(data: data) {
                    calibrationBackgroundImageArray.append(uiImage)
                } else {
                    print("❌ Failed to load background image: \(filename)")
                }
            }
            print("✅ calibrationBackgroundImageArray Load Complete, count: \(calibrationBackgroundImageArray.count)")
        } catch {
            print("❌ Error loading background dataset from local storage: \(error)")
        }
    }
}
