# EyeTrackingWork

## Troubleshooting

### `pod install` fails with `unknown ISA 'PBXFileSystemSynchronizedRootGroup'`

Xcode 16 introduced synchronized folder groups, which older versions of the `xcodeproj` gem can't parse. Upgrade CocoaPods and `xcodeproj`:

```bash
gem install cocoapods
gem install xcodeproj
```

(Use `sudo` if your gem directory requires it.) You need `xcodeproj` ≥ 1.25.0 and CocoaPods ≥ 1.16. Verify with:

```bash
pod --version
gem list xcodeproj
```

Then re-run `pod install`.

Alternative: in Xcode, right-click the synchronized folder in the navigator and choose **Convert to Group** to fall back to a classic `PBXGroup`.