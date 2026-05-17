//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//
import Foundation
import Combine
import CoreGraphics

class TrialState: ObservableObject {
    static let shared = TrialState()
    @Published var trialStarted: Bool = false
    @Published var currentTrialNum: Int = 1     // 1-based indexing
    @Published var currentBackgroundIndex: Int = 0
    @Published var calibrationCurrentBackgroundIndex: Int = 0   // used only for calibration data collection session
    @Published var trialReviewCompleted: Bool = true
}


struct TrialLogEntry: Codable {
    let subject: String
    let session: String
    let trial: Int
    let px_x: CGFloat
    let px_y: CGFloat
    let px_norm_x: CGFloat
    let px_norm_y: CGFloat
    let background: String
    let frameNum: Int
    let time: TimeInterval
}

class TrialLogger: ObservableObject {
    static let shared = TrialLogger()
    
    private var startTime: Date = Date()
    @Published var logEntries: [TrialLogEntry] = []

    func startSession() {
        logEntries = []
        startTime = Date()
    }

    func logTrial(subject: String, session: String, trial: Int, position: CGPoint, background: String) {
        let normalizedX = position.x / 1000.0
        let normalizedY = position.y / 1000.0
        let elapsed = Date().timeIntervalSince(startTime)
        
        let entry = TrialLogEntry(
            subject: subject,
            session: session,
            trial: trial,
            px_x: position.x,
            px_y: position.y,
            px_norm_x: normalizedX,
            px_norm_y: normalizedY,
            background: background,
            frameNum: trial,
            time: elapsed
        )
        logEntries.append(entry)
//        print("log added: \(entry.subject),\(entry.session),\(entry.trial),\(entry.px_x),\(entry.px_y),\(entry.px_norm_x),\(entry.px_norm_y),\(entry.background),\(entry.frameNum),\(entry.time)")
    }

    func removeLastTrial() {
        if !logEntries.isEmpty {
            logEntries.removeLast()
        }
    }

    func exportCSV(subject: String, session: String) {
        let filename = "p\(subject)_s\(session)_study2_data.csv"
            let baseURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let subjectFolder = baseURL.appendingPathComponent("rawdata").appendingPathComponent("p\(subject)")
            let fileURL = subjectFolder.appendingPathComponent(filename)
            
        let header = "subject,session,trial,px_x,px_y,px_norm_x,px_norm_y,background,frameNum,time\n"
        var csvText = header
        for entry in logEntries {
            csvText += "\(entry.subject),\(entry.session),\(entry.trial),\(entry.px_x),\(entry.px_y),\(entry.px_norm_x),\(entry.px_norm_y),\(entry.background),\(entry.frameNum),\(entry.time)\n"
        }
        
        do {
            try FileManager.default.createDirectory(at: subjectFolder, withIntermediateDirectories: true, attributes: nil)
            try csvText.write(to: fileURL, atomically: true, encoding: .utf8)
            print("✅ CSV saved at \(fileURL.path)")
        } catch {
            print("❌ Failed to save CSV: \(error)")
        }
    }
}
