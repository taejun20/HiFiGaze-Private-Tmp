import Foundation
import AVFoundation
import CoreImage
import UIKit

final class DataRecorder {
    private(set) var subject: String = ""
    private(set) var screenName: String = ""

    private let ciContext = CIContext()
    private var sessionStartTime: CFTimeInterval = 0

    private var documentsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var framesDir: URL {
        documentsURL.appendingPathComponent("p\(subject)/frames", isDirectory: true)
    }

    private var csvDir: URL {
        documentsURL.appendingPathComponent("p\(subject)", isDirectory: true)
    }

    private(set) var nextFrameIndex: Int = 1
    private(set) var rows: [CSVRow] = []
    private var savedFrameURLsForCurrentTrial: [URL] = []
    private var rowCountAtTrialStart: Int = 0

    var isRecording: Bool = false
    var currentGT: CGPoint = .zero

    func configure(subject: String, screenName: String) {
        self.subject = subject
        self.screenName = screenName
        prepareFolders()
    }

    func startSessionClock() {
        sessionStartTime = CACurrentMediaTime()
    }

    func beginTrialRecordingContext() {
        savedFrameURLsForCurrentTrial.removeAll()
        rowCountAtTrialStart = rows.count
    }

    func commitTrial() {
        savedFrameURLsForCurrentTrial.removeAll()
    }

    func discardTrial() {
        // delete frames saved this trial
        for url in savedFrameURLsForCurrentTrial {
            try? FileManager.default.removeItem(at: url)
        }
        savedFrameURLsForCurrentTrial.removeAll()

        // truncate rows
        if rowCountAtTrialStart <= rows.count {
            rows.removeSubrange(rowCountAtTrialStart..<rows.count)
        }
    }

    func finalizeCSV() {
        let fileURL = csvDir.appendingPathComponent("data.csv")
        var text = "subject,frameID,screen,gt_x_px,gt_y_px,timestamp\n"
        for r in rows {
            text += "\(r.subject),\(r.frameID),\(r.screen),\(r.gt_x_px),\(r.gt_y_px),\(String(format: "%.6f", r.timestamp))\n"
        }
        try? text.write(to: fileURL, atomically: true, encoding: .utf8)
    }

    func handleSampleBuffer(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let elapsed = CACurrentMediaTime() - sessionStartTime
        let frameID = String(format: "%05d", nextFrameIndex)
        nextFrameIndex += 1

        let x = Int(currentGT.x.rounded())
        let y = Int(currentGT.y.rounded())

        // Convert to JPEG
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        let uiImage = UIImage(cgImage: cgImage)

        guard let jpegData = uiImage.jpegData(compressionQuality: 0.92) else { return }

        let outURL = framesDir.appendingPathComponent("\(frameID).jpg")
        do {
            try jpegData.write(to: outURL, options: .atomic)
            savedFrameURLsForCurrentTrial.append(outURL)
        } catch {
            return
        }

        rows.append(
            CSVRow(
                subject: subject,
                frameID: "\(frameID).jpg",
                screen: screenName,
                gt_x_px: x,
                gt_y_px: y,
                timestamp: elapsed
            )
        )
    }

    // MARK: - Helpers

    private func prepareFolders() {
        try? FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true)
        try? FileManager.default.createDirectory(at: csvDir, withIntermediateDirectories: true)
    }
}
