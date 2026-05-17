//
//  UDPClient.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import Foundation
import Network
import SwiftUI
import Combine

class TCPClient: ObservableObject {
    static let shared = TCPClient()
    var subject: String = ""
    var session: String = ""
    
    @Published var eyeImageLeft: UIImage? = nil
    @Published var eyeImageRight: UIImage? = nil
    
    private var connection: NWConnection?
    private let queue = DispatchQueue(label: "TCPClientQueue")

    let serverHost = NWEndpoint.Host("172.20.10.4") // e.g., "192.168.1.100"
    let serverPort: NWEndpoint.Port = 12345
    
    private var receiveBuffer = Data()
    private let messageDelimiter = "\n"
    
    @Published var receivedFocusValue: Int = 210

    private init() {}
    
    func connect(completion: @escaping (Bool) -> Void) {
        connection = NWConnection(host: serverHost, port: serverPort, using: .tcp)

        connection?.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready:
                print("✅ TCP connected")
                self?.startReceiving()
                completion(true)
            case .failed(let error):
                print("❌ TCP connection failed: \(error.localizedDescription)")
                self?.connection = nil
                completion(false)
            case .cancelled:
                print("ℹ️ TCP connection cancelled")
                self?.connection = nil
                completion(false)
            default:
                break
            }
        }

        connection?.start(queue: queue)
    }

    func send(message: String) {
//        print("📝 send: \(message)")
        guard let connection = connection else { return }
        
        let data = message.data(using: .utf8) ?? Data()
        connection.send(content: data, completion: .contentProcessed({ sendError in
            if let sendError = sendError {
                print("❌ Send error: \(sendError.localizedDescription)")
                return
            }
        }))
    }
    
    private func startReceiving() {
        guard let connection = connection else { return }

        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, isComplete, error in
            guard let strongSelf = self else { return }

            if let error = error {
                print("❌ Receive error: \(error.localizedDescription)")
                return
            }
            
            if let data = data {
                strongSelf.receiveBuffer.append(data)

                while let delimiterRange = strongSelf.receiveBuffer.range(of: Data(strongSelf.messageDelimiter.utf8)) {
                    let messageData = strongSelf.receiveBuffer.subdata(in: 0..<delimiterRange.lowerBound)
                    strongSelf.receiveBuffer.removeSubrange(0..<delimiterRange.upperBound) // Remove delimiter

                    if let message = String(data: messageData, encoding: .utf8) {
                        strongSelf.handleMessage(message)
                    } else {
                        print("❌ Failed to decode message as UTF-8")
                    }
                }
            }

            // Keep receiving
            strongSelf.startReceiving()
        }
    }
    
    private func handleMessage(_ message: String) {
        if message.hasPrefix("text:") {
            let text = message.replacingOccurrences(of: "text:", with: "")
//            print("📝 Received text: \(text)")

            if text == "trialfin" {
                DispatchQueue.main.async {
                    let trialState = TrialState.shared
                    trialState.trialReviewCompleted = false
                    trialState.trialStarted = false
                    trialState.currentTrialNum += 1
                    trialState.currentBackgroundIndex += 1
                    trialState.calibrationCurrentBackgroundIndex = trialState.currentBackgroundIndex / 5
//                    print("🔁 trialfin: advanced to trial \(trialState.currentTrialNum)")
                }
            }
            else if text == "complete" {
                let trialState = TrialState.shared
                if !trialState.trialReviewCompleted {
                    send(message: "trialreviewfin:p\(subject)s\(session)t\(trialState.currentTrialNum-1)\n")  // to let the python PC save the frame
                    DispatchQueue.main.async {
                        trialState.trialReviewCompleted = true
                    }
                }
            }
            else if text == "redo" {
                let trialState = TrialState.shared
                if !trialState.trialReviewCompleted {
                    DispatchQueue.main.async {
                        TrialLogger.shared.removeLastTrial()
                        trialState.currentTrialNum -= 1
                        trialState.currentBackgroundIndex -= 1
                        trialState.calibrationCurrentBackgroundIndex = trialState.currentBackgroundIndex / 5
                        trialState.trialReviewCompleted = true
                    }
                }
            }
            else if text.hasPrefix("focus") {
                let valueString = String(text.dropFirst("focus".count))
                if let focusValue = Int(valueString) {
                    DispatchQueue.main.async { [weak self] in
                        self?.receivedFocusValue = focusValue   // 👉 publish to UI
                    }
                }
            }
        } else if message.hasPrefix("image:") {
            let imagePayload = message.replacingOccurrences(of: "image:", with: "")
            let components = imagePayload.components(separatedBy: "|")
            if components.count == 2,
               let leftData = Data(base64Encoded: components[0]),
               let rightData = Data(base64Encoded: components[1]),
               let leftImage = UIImage(data: leftData),
               let rightImage = UIImage(data: rightData) {

                DispatchQueue.main.async {
                    self.eyeImageLeft = leftImage
                    self.eyeImageRight = rightImage
                }
            } else {
                print("❌ Failed to decode eye images")
            }
        }
    }
    
    func disconnect() {
        connection?.cancel()
        connection = nil
        print("🔌 Disconnected from TCP server")
    }
}
