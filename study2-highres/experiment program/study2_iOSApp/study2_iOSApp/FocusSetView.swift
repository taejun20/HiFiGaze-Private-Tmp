//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import SwiftUI

struct FocusSetView: View {    
    @Binding var focusSelected: Bool
    @State private var focusValue: Double = -1
    @ObservedObject private var tcpClient = TCPClient.shared

    var body: some View {
        GeometryReader { geometry in
            let imageWidth = geometry.size.width
            let imageHeight = imageWidth / 2
            
            VStack {
                Text("Focus: \(tcpClient.receivedFocusValue)")
                    .font(.title)

                // Complete button
                Button("Set autofocus") {
                    TrueDepthManager.shared.startOnce()
                }
                .frame(width: 300, height: 60)
                .font(.largeTitle)
                .background(Color.white)
                .cornerRadius(10)
                .padding(.top, 40)

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
                Button("Complete") {
                    TCPClient.shared.send(message: "setfocusfin\n")
                    focusSelected = true
                }
                .frame(width: 200, height: 70)
                .font(.largeTitle)
                .background(Color.white)
                .cornerRadius(10)
                .padding(.top, 40)
            }
        }
        .onAppear {
            TCPClient.shared.send(message: "setfocus\n")
        }
    }
}
