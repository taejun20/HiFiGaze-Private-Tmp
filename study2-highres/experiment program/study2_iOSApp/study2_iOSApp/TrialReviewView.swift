//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import SwiftUI

struct TrialReviewView: View {
    @Binding var focusSelected: Bool
    let subject: String
    let session: String
    
    let eyeImageSizeGain: CGFloat = 0.95

    @ObservedObject private var tcpClient = TCPClient.shared
    @ObservedObject var trialState = TrialState.shared

    var body: some View {
        GeometryReader { geometry in
            let imageWidth = geometry.size.width * eyeImageSizeGain
            let imageHeight = imageWidth / 2 * eyeImageSizeGain
            
            VStack (spacing: 20) {
                // Show current focus value
                Button("Set Focus") {
                    focusSelected = false
                }
                .font(.largeTitle)
                .padding()
                
//                Button("Redo") {
//                    TrialLogger.shared.removeLastTrial()
//                    trialState.currentTrialNum -= 1
//                    trialState.currentBackgroundIndex -= 1
//                    trialState.calibrationCurrentBackgroundIndex = trialState.currentBackgroundIndex / 5
//                    trialState.trialReviewCompleted = true
//                }
//                .frame(width: 160, height: 50)
//                .font(.largeTitle)
//                .background(Color.white)
//                .cornerRadius(10)
//                .padding(.top, 10)
                
                if let right = tcpClient.eyeImageRight {
                    Image(uiImage: right)
                        .resizable()
                        .frame(width: imageWidth, height: imageHeight)
                } else {
                    Color.gray
                        .frame(width: imageWidth, height: imageHeight)
                }
                
                // Eye images
                if let left = tcpClient.eyeImageLeft {
                    Image(uiImage: left)
                        .resizable()
                        .frame(width: imageWidth, height: imageHeight)
                } else {
                    Color.gray
                        .frame(width: imageWidth, height: imageHeight)
                }
                // Complete button
//                Button("Complete Complete") {
//                    TCPClient.shared.send(message: "trialreviewfin:p\(subject)s\(session)t\(trialState.currentTrialNum-1)\n")  // to let the python PC save the frame
//                    trialState.trialReviewCompleted = true
//                }
//                .frame(width: 400)
//                .font(.largeTitle)
//                .padding()
//                .background(Color.white)
//                .cornerRadius(10)
                
            }
        }
    }
}
