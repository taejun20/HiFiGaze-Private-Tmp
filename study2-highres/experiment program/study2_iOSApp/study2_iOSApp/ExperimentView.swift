//
//  ContentView.swift
//  study2_iOSApp
//
//  Created by TJ on 8/5/25.
//

import SwiftUI

struct ExperimentView: View {
    let subject: String
    let session: String
    let totalTrialNum: Int
    @ObservedObject var trialState = TrialState.shared
    @Binding var trialTestOrderArray: [Int]
    @Binding var backgroundImageArray: [UIImage]
    @Binding var backgroundImageFilenameArray: [String]
    let dotRadius: CGFloat

    let horizontalLaneCount: Int
    let verticalLaneCount: Int
    
    let trialDuration: TimeInterval  // <--- Add this line

    @State private var dotStartPoint = CGPoint.zero

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                //Image(uiImage: backgroundImageArray[trialState.currentBackgroundIndex])
                Image(uiImage: backgroundImageArray[0])     // test
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
        if trial <= totalTrialNum
        {
            //let positionIndex = trialTestOrderArray[trial-1]
            let positionIndex = 2
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
            ret = CGPoint(x: x, y: y)
        }
        return ret
    }
}
