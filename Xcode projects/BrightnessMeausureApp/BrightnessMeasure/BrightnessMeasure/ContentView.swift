//
//  ContentView.swift
//  BrightnessMeasure
//
//  Created by TJ on 7/16/25.
//

import SwiftUI

struct ContentView: View {
    @State private var brightness: CGFloat = UIScreen.main.brightness

    var body: some View {
        VStack(spacing: 20) {
            Text("Current Brightness")
                .font(.headline)
            Text(String(format: "%.3f", brightness))
                .font(.largeTitle)
            Button("Refresh") {
                brightness = UIScreen.main.brightness
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
