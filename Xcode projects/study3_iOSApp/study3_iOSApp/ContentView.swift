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
    
    let totalTrialNum = 14
    @State private var subjectSelected = false
    @State private var sbjectSelected = false
    @State private var sessionSelected = false
    @State private var trialStarted = false
    @State private var currentTrialNum = 1  // 1-based indexing
    let horizontalLaneCount = 9
    let verticalLaneCount = 5
    let dotRadius: CGFloat = 10
    @State var trialTestOrderArray: [String] = []
    
    @State private var appStartTime: Double = CACurrentMediaTime()
    @State private var expFinished = false
    @State private var backgroundImageArray: [UIImage] = []  // Load all images into memory once
    @State private var backgroundImageFilenameArray: [String] = []
    @State private var currentBackgroundIndex = 0
    @State private var nextBackgroundIndex = 0
    @StateObject private var camera = CameraController()
    private let countdownTimer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    @State private var countdown: Double = 5.0

    @State private var isCalibration = false
    @State private var calibrationStarted = false
    @State private var calibrationFinished = false
    @State private var calibrationCurrentTrialNum = 1   // 1-based indexing
    @State private var calibrationTotalTrialNum = 15
    @State private var calibrationBackgroundImageArray: [UIImage] = []  // Load all images into memory once
    @State private var calibrationCurrentBackgroundIndex = 0

    var body: some View {
        ZStack {
            if subjectSelected {
                if sessionSelected {
                    if !isCalibration {
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
                                    backgroundImageArray: $backgroundImageArray,
                                    backgroundImageFilenameArray: $backgroundImageFilenameArray,
                                    currentBackgroundIndex: $currentBackgroundIndex,
                                    nextBackgroundIndex: $nextBackgroundIndex,
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

                                        Button("Start Start Start Start Start") {
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
                        if calibrationFinished
                        {
                            VStack(spacing: 40) {
                                Text("p\(subject), session: \(session)")
                                    .font(.title)
                                                                
                                Text("✅ Calibration Finished")
                                    .font(.largeTitle)
                                    .foregroundColor(.green)
                            }
                            .onAppear() {
                                saveCSVCalibration()
                            }
                        }
                        else
                        {
                            if calibrationStarted
                            {
                                CalibrationView(
                                    subject: subject,
                                    session: session,
                                    calibrationCurrentTrialNum: $calibrationCurrentTrialNum,
                                    calibrationTotalTrialNum: calibrationTotalTrialNum,
                                    calibrationStarted: $calibrationStarted,
                                    calibrationBackgroundImageArray: $calibrationBackgroundImageArray,
                                    camera: camera,
                                    dotRadius: dotRadius,
                                    appStartTime: appStartTime,
                                    calibrationCurrentBackgroundIndex: $calibrationCurrentBackgroundIndex,
                                    calibrationFinished: $calibrationFinished
                                )
                            }
                            else
                            {
                                ZStack {
                                    VStack (spacing: 20) {
                                        Text("p\(subject), session: \(session)")
                                            .font(.title)
                                        
                                        Text("Calibration \(calibrationCurrentTrialNum) / \(calibrationTotalTrialNum)")
                                            .font(.title)

                                        Button("Start Start Start Start Start") {
                                            calibrationStarted = true
                                        }
                                        .font(.largeTitle)
                                        .padding()
                                    }
                                    Circle()
                                        .fill(Color.gray)
                                        .frame(width: dotRadius * 2, height: dotRadius * 2)
                                        .position(getInitialDotPosition(trial: calibrationCurrentTrialNum))
                                    
                                    Circle()
                                        .fill(Color.white)
                                        .frame(width: dotRadius * 0.4, height: dotRadius * 0.4)
                                        .position(getInitialDotPosition(trial: calibrationCurrentTrialNum))
                                }
                            }
                        }
                    }
                }
                else
                {
                    VStack(spacing: 45) {
                        HStack(spacing: 60) {
                            Button("prac-1") {
                                if !sessionSelected {   // handling quick double tap case
                                    sessionSelected = true
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "practice-1"
                                    loadBackgroundDataset()
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("prac-2") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "practice-2"
                                    loadBackgroundDataset()
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("s1-1") {
                                if !sessionSelected {   // handling quick double tap case
                                    sessionSelected = true
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "1p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "1s"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "2p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "2s"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "3p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "3s"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "4p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "4s"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "5p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "5s"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)" } + (0...4).map { "v\($0)" }
                                    session = "6p"
                                    loadBackgroundDataset()
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
                                    trialTestOrderArray = (0...8).map { "h\($0)r" } + (0...4).map { "v\($0)r" }
                                    session = "6s"
                                    loadBackgroundDataset()
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 100)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                        }
                        HStack(spacing: 60) {
                            Button("calsit") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    isCalibration = true
                                    camera.isCalibration = true
                                    session = "calsit"
                                    loadBackgroundDatasetCalibration()
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 130)
                            .font(.largeTitle)
                            .background(Color.white)
                            .cornerRadius(10)
                            Button("calstand") {
                                if !sessionSelected {
                                    sessionSelected = true
                                    isCalibration = true
                                    camera.isCalibration = true
                                    session = "calstand"
                                    loadBackgroundDatasetCalibration()
                                    camera.configure(subject: subject, session: session)
                                }
                            }
                            .frame(width: 150)
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
        .rotationEffect(.degrees(180), anchor: .center)
        .onAppear() {
            camera.appStartTime = appStartTime
            camera.startSession()
        }
        .onDisappear {
            camera.stopSession()
        }
    }
    
    func getInitialDotPosition(trial: Int) -> CGPoint {
        if !isCalibration {
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
    
    func redoLastTrial() {
        currentTrialNum -= 1
        camera.clearTrialData(deletingTrialNum: currentTrialNum)
        trialStarted = true
    }

    func loadBackgroundDataset() {
        backgroundImageFilenameArray.removeAll()
        backgroundImageArray.removeAll()
        let screenshotDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("backgrounds-study3/p\(subject)/\(session)")
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
            let calibrationBackgroundImageFilenameArray = files
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
            print("❌ Error loading calibration background dataset from local storage: \(error)")
        }
    }
    
    func saveCSV() {
        camera.saveUIStateLogs()
    }
    
    func saveCSVCalibration() {
        camera.saveUIStateLogsCalibration()
    }
}
