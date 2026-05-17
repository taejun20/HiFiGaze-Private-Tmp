//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import SwiftUI

struct CalibrationView: View {
    let subject: String
    let session: String
    let calibrationTotalTrialNum: Int
    @ObservedObject var trialState = TrialState.shared
    @Binding var calibrationBackgroundImageArray: [UIImage]
    let dotRadius: CGFloat

    let horizontalLaneCount: Int
    let verticalLaneCount: Int
    
    let trialDuration: TimeInterval  // <--- Add this line

    @State private var dotStartPoint = CGPoint.zero

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Image(uiImage: calibrationBackgroundImageArray[trialState.calibrationCurrentBackgroundIndex == 3 ? 2 : trialState.calibrationCurrentBackgroundIndex])
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
                .position(getDotPosition(trial: trialState.currentTrialNum))
            }
        }
    }
    
    func getDotPosition(trial: Int) -> CGPoint {
        var ret = CGPoint.zero
        if trial <= calibrationTotalTrialNum
        {
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
        }
        return ret
    }
}
