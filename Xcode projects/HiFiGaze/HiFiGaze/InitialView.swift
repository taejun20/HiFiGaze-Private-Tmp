import SwiftUI

struct InitialView: View {
    @State private var showMainView = false
    @State private var showCalibrationView = false
    @State private var showRGB = false
    @State private var showRGBT = true

    @State private var calibrationPointCount = 5
    @State private var calibrationSamples: [GazeSample] = []
    @State private var screenBrightness: CGFloat = UIScreen.main.brightness
    @State private var showBrightnessWarning = false
    @State private var shrinkDuration: Double = 1.5
    
    private let minBrightness: CGFloat = 0.25
    var body: some View {
        if showMainView {
            MainView(
                showRGB: showRGB,
                showRGBT: showRGBT,
                gazeSamples: calibrationSamples,
                onRecalibrate: {
                    calibrationSamples = []
                    showMainView = false
                    showCalibrationView = true
                }
            )
        }
        else if showCalibrationView {
            CalibrationView(
                showRGB: showRGB,
                showRGBT: showRGBT,
                calibrationPointCount: calibrationPointCount,
                shrinkDuration: shrinkDuration
            ) { samples in
                calibrationSamples = samples
                showCalibrationView = false
                showMainView = true
            }
            //MainViewForDebug(showRGB: showRGB, showRGBT: showRGBT)
            //MainViewForDebug_Socket(showRGB: showRGB, showRGBT: showRGBT)
            //EyeCropProcessView()
        } else {
            ZStack {
                Color.black
                    .ignoresSafeArea()

                VStack(spacing: 20) {
                    VStack(spacing: 10) {
                        Text(calibrationPointCount == 10 ? "10-point" : "5-point")
                            .font(.system(size: 22, weight: .semibold))
                            .foregroundColor(.white)

                        Toggle("", isOn: Binding(
                            get: { calibrationPointCount == 5 },
                            set: { calibrationPointCount = $0 ? 5 : 10 }
                        ))
                        .labelsHidden()
                        .toggleStyle(SwitchToggleStyle(tint: .gray))
                        .scaleEffect(1.3)
                    }
                    .padding(.horizontal, 30)
                    .padding(.vertical, 20)
                    .background(Color.white.opacity(0.12))
                    .cornerRadius(20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
                    .padding(.top, 180)
                    
                    
                    
                    VStack(spacing: 10) {
                        Text(String(format: "Fixating Duration: %.1f s", shrinkDuration))
                            .font(.system(size: 20, weight: .semibold))
                            .foregroundColor(.white)

                        Slider(value: $shrinkDuration, in: 1.5...10.0, step: 0.5)
                            .tint(.green)

                        HStack {
                            Text("1.5s")
                            Spacer()
                            Text("10s")
                        }
                        .font(.system(size: 14))
                        .foregroundColor(.white.opacity(0.7))
                    }
                    .padding(.horizontal, 30)
                    .padding(.vertical, 18)
                    .background(Color.white.opacity(0.12))
                    .cornerRadius(20)
                    .overlay(
                        RoundedRectangle(cornerRadius: 20)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
                    .padding(.horizontal, 30)
                    
                    
                    // Title
                    Text("HiFiGaze")
                        .font(.system(size: 50))
                        .fontWeight(.thin)
                        .foregroundColor(.white)
                        .padding(.top, 40)

                    // Checkboxes (Custom)
//                    VStack(alignment: .leading, spacing: 20) {
//                        CustomCheckbox(isOn: $showRGB, label: "RGB")
//                        CustomCheckbox(isOn: $showRGBT, label: "RGB + Screen Knowledge")
//                    }
//                    .padding(.horizontal, 10)
//                    .font(.title2)

                    
                    
                    // Start button
                    Button(action: {
                       if screenBrightness < minBrightness {
                           showBrightnessWarning = true
                       } else {
                           showBrightnessWarning = false
                           withAnimation(.easeInOut) {
                               showCalibrationView = true
                           }
                       }
                    }) {
                        Text("Start")
                            .font(.system(size: 40))
                            .fontWeight(.regular)
                            .foregroundColor(.white)
                            .padding()
                            .frame(width: 200, height: 100)
                            .background(Color.blue)
                            .cornerRadius(15)
                            .shadow(radius: 5)
                    }

                    Spacer()
                }
                if showBrightnessWarning && screenBrightness < minBrightness {
                    Text("Raise screen brightness")
                        .font(.system(size: 30, weight: .semibold))
                        .foregroundColor(.black)
                        .padding(14)
                        .background(Color.white)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                        .multilineTextAlignment(.center)
                        .offset(y: -220)
                }
            }
            .onAppear {
                screenBrightness = UIScreen.main.brightness
            }
            .onReceive(NotificationCenter.default.publisher(
                for: UIScreen.brightnessDidChangeNotification
            )) { _ in
                screenBrightness = UIScreen.main.brightness

                if screenBrightness >= minBrightness {
                    showBrightnessWarning = false
                }
            }
        }
    }
}

/// ✅ Custom Checkbox Toggle for iOS
struct CustomCheckbox: View {
    @Binding var isOn: Bool
    var label: String

    var body: some View {
        Button(action: {
            isOn.toggle()
        }) {
            HStack {
                Image(systemName: isOn ? "checkmark.square.fill" : "square")
                    .foregroundColor(isOn ? .blue : .gray)
                    .font(.title2)
                Text(label)
                    .foregroundColor(.black)
            }
        }
        .buttonStyle(.plain)
    }
}

#Preview {
    InitialView()
}
